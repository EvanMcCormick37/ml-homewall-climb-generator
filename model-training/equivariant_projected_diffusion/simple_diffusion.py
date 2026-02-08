import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
from climb_conversion import ClimbsFeatureScaler
import sqlite3
import math

# Input Data Format:
# climbs: Tensor [Batch_length, 20, 5]
# conditions: Tensor [Batch_length, 4]
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=padding)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

        self.cond_proj = nn.Linear(cond_dim, out_channels*2)
        self.shortcut = nn.Conv1d(in_channels,out_channels,1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(x)
        h = self.norm1(h)

        gamma, beta = self.cond_proj(cond).unsqueeze(-1).chunk(2, dim=1)
        h = h*(1+gamma) + beta

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.shortcut(x)

class Denoiser(nn.Module):
    def __init__(self, hidden_dim=64, layers = 4):
        super().__init__()

        self.cond_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.init_conv = ResidualBlock1D(4, hidden_dim, hidden_dim)

        self.residuals = nn.ModuleList([nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU()
        ) for _ in range(layers)])

        self.head = nn.Conv1d(hidden_dim, 4, 1)
    
    def forward(self, climbs: Tensor, cond: Tensor, t: Tensor)-> Tensor:
        """
        Run denoising pass. Predicts the denoised dataset from the noisy data.
        
        :param climbs: Tensor with hold-set positions. [B, S, 4]
        :param cond: Tensor with conditional variables. [B, 4]
        :param t: Tensor with timestep of diffusion. [B, 1]
        """
        emb_c = self.cond_mlp(torch.cat([cond,t], dim=1))
        emb_h = self.init_conv(climbs.transpose(1,2), emb_c)

        for layer in self.residuals:
            emb_h = emb_h + layer(emb_h)


        result = self.head(emb_h).transpose(1,2)

        return result

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Standard sinusoidal positional embeddings for time steps.
    Helps the model understand 'where' it is in the diffusion process.
    """
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time: (batch_size, 1)
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, 0].unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AttentionBlock1D(nn.Module):
    """
    Self-Attention block for 1D sequence data.
    Allows every hold to 'attend' to every other hold, capturing global dependencies.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        
    def forward(self, x):
        # x: (Batch, Channels, Length)
        B, C, L = x.shape
        
        # Norm and rearrange for Attention: (Batch, Length, Channels)
        h = self.norm(x).permute(0, 2, 1)
        
        # Self-Attention
        # attn_output: (Batch, Length, Channels)
        attn_output, _ = self.attention(h, h, h)
        
        # Rearrange back to (Batch, Channels, Length)
        h = attn_output.permute(0, 2, 1)
        
        # Residual connection
        return x + h

class Noiser(nn.Module):
    def __init__(self, hidden_dim=128, layers = 5, feature_dim = 4, cond_dim = 4, sinusoidal = True):
        super().__init__()

        self.time_mlp = nn.Sequential(
            (SinusoidalPositionEmbeddings(hidden_dim) if sinusoidal else nn.Linear(1,hidden_dim)),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim)
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.combine_t_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.init_conv = ResidualBlock1D(feature_dim, hidden_dim, hidden_dim)

        self.down_blocks = nn.ModuleList([ResidualBlock1D(hidden_dim*(i+1), hidden_dim*(i+2), hidden_dim) for i in range(layers)])
        self.up_blocks = nn.ModuleList([ResidualBlock1D(hidden_dim*(i+1), hidden_dim*(i), hidden_dim) for i in range(layers,0,-1)])

        self.head = nn.Conv1d(hidden_dim, feature_dim, 1)
    
    def forward(self, climbs: Tensor, cond: Tensor, t: Tensor)-> Tensor:
        """
        Run denoising pass. Predicts the added noise from the noisy data.
        
        :param climbs: Tensor with hold-set positions. [B, S, 4]
        :param cond: Tensor with conditional variables. [B, 4]
        :param t: Tensor with timestep of diffusion. [B, 1]
        """
        emb_t = self.time_mlp(t)
        emb_c = self.cond_mlp(cond)
        emb_c = self.combine_t_mlp(torch.cat([emb_t, emb_c],dim=1))
        emb_h = self.init_conv(climbs.transpose(1,2), emb_c)
        
        residuals = []
        for layer in self.down_blocks:
            residuals.append(emb_h)
            emb_h = layer(emb_h, emb_c)
        
        for layer in self.up_blocks:
            residual = residuals.pop()
            emb_h = residual + layer(emb_h, emb_c)
        
        result = self.head(emb_h).transpose(1,2)

        return result

class ClimbDDPM(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.timesteps = 100
    
    def _cos_alpha_bar(self, t: Tensor)-> Tensor:
        t = t.view(-1,1,1)
        epsilon = 0.0004
        return  torch.cos((t+epsilon)/(1+2*epsilon)*torch.pi/2)**2
    
    def loss(self, sample_climbs, cond):
        """Perform a diffusion Training step and return the loss resulting from the model in the training run. Currently returns tuple (loss, real_hold_loss, null_hold_loss)"""
        B, S, H = sample_climbs.shape
        t = torch.round(torch.rand(B, 1, device=self.device), decimals=2)
        x_0 = torch.randn((B, S, H), device = self.device)
        noisy = self.forward_diffusion(sample_climbs, t, x_0)
        pred_x_0 = self.model(noisy, cond, t)
        return F.mse_loss(pred_x_0, x_0)
    
    def predict_clean(self, noisy, cond, t):
        """Return predicted clean data."""
        a = self._cos_alpha_bar(t)
        prediction = self.model(noisy, cond, t)
        clean = (noisy - torch.sqrt(1-a)*prediction)/torch.sqrt(a)
        return clean
    
    def forward_diffusion(self, clean: Tensor, t: Tensor, x_0: Tensor)-> Tensor:
        """Perform forward diffusion to add noise to clean data based on noise adding schedule."""
        a = self._cos_alpha_bar(t)
        return torch.sqrt(a) * clean + torch.sqrt(1-a) * x_0
    
    def forward(self, noisy, cond, t):
        return self.predict_clean(noisy, cond, t)

class ClimbDASD(nn.Module):
    def __init__(self, hidden_dim=128, layers=3, sinusoidal=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timesteps = 100
        self.model = Noiser(hidden_dim,layers, feature_dim=9, sinusoidal=sinusoidal)
        self.roles_head = nn.Softmax(dim=2)

    def _cos_alpha_bar(self, t: Tensor)-> Tensor:
        t = t.view(-1,1,1)
        epsilon = 0.0004
        return  torch.cos((t+epsilon)/(1+2*epsilon)*torch.pi/2)**2
        
    def predict(self, noisy, cond, t):
        """Return prediction for noise."""
        pred = self.model(noisy, cond, t)
        pred_noise = pred[:,:,:4]
        pred_roles = self.discrete_features_head(pred[:,:,4:])

        a = self._cos_alpha_bar(t)

        pred_clean_cont = (noisy - torch.sqrt(1-a)*pred_noise)/torch.sqrt(a)

        return pred_clean_cont, pred_roles
    
    def loss(self, clean, cond):
        """Get the model's loss from training on a dataset of clean (denoised) data."""
        B, H, S = clean.shape
        t = torch.round(torch.rand(B, 1, device=self.device), decimals=2)
        x_0 = torch.randn((B, H, S-5))
        noisy = self.forward_diffusion(clean, t, x_0)

        pred = self.model(noisy, cond, t)
        pred_roles = self.roles_head(pred[:,:,4:])
        continuous_loss = F.mse_loss(pred[:,:,:4], x_0[:,:,:4])
        discrete_loss = F.cross_entropy(pred_roles, clean[:,:,4:])

        return continuous_loss + discrete_loss, continuous_loss, discrete_loss
    
    def forward_diffusion(self, clean, t, x_0: Tensor):
        """
        Perform the forward Diffusion process in two stages:
            *Continuous Diffusion over Continuous Features [0:4]
            *Discrete Absorbing State Diffusion over Discrete Features [4:9]
        
        :param clean: Full feature set (4 continuous, 5 discrete roles OH-Encoded)
        :param t: Timestep Tensor
        :param x_0: Optionally include the tensor for the prior 'full-noise' array to use instead of random noise. This makes the generation process deterministic.
        :return: Diffused climbs, conditioned on timestep
        :rtype: Tensor
        """
        cont_feat = clean[:,:,:4]
        disc_feat = clean[:,:,4:]
        a = self._cos_alpha_bar(t)

        diff_cont = torch.sqrt(a)*cont_feat + torch.sqrt(1-a)*(x_0)

        das_mask = (a > torch.rand_like(a)).float()
        diff_disc = disc_feat*das_mask

        return torch.cat([diff_cont,diff_disc],dim=2)
    
    def forward(self, noisy, cond, t):
        return self.predict(noisy, cond, t)

class DDPMTrainer():
    def __init__(
        self,
        model: nn.Module,
        dataset: TensorDataset | None = None,
        default_batch_size: int = 64,
        lr: float = 1e-3
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataset = dataset
        self.default_batch_size = default_batch_size
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
    
    def train(
        self,
        epochs: int,
        save_path: str,
        batch_size: int | None = None,
        num_workers: int | None = None,
        dataset: TensorDataset | None = None,
        save_on_best: bool = False,
        clip_grad_norm: bool = True
    )-> tuple[nn.Module, list]:
        """
        Train a model (probably of type ClimbDDPM) on the dataset contained in the trainer. (If dataset is provided, train on that dataset instead)

        :param epochs: Number of training epochs
        :type epochs: int
        :param save_path: Model weights save-path
        :type save_path: str
        :param batch_size: Training batch size
        :type batch_size: int | None
        :param num_workers: Number of workers
        :type num_workers: int | None
        :param dataset: Training Dataset; defaults to model.dataset
        :type dataset: TensorDataset | None
        :param save_on_best: boolean indicating whether to save model weights every time a minimum loss is reached.
        :type save_on_best: bool
        :return: Tuple of (best_model: nn.Module, training_data: np.array)
        :rtype: tuple[Module, Any]
        """
        if dataset is None:
            dataset = self.dataset
        if dataset is None:
            raise ValueError("Dataset is None. Cannot train on no dataset")
        if batch_size is None:
            batch_size = self.default_batch_size
        if num_workers is None:
            num_workers = 0

        batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        losses = []

        with tqdm(range(epochs)) as pbar:
            for epoch in pbar:
                total_loss = 0
                for x, c in batches:
                    x, c = x.to(self.device), c.to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.model.loss(x, c)
                    if clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    loss.backward()
                    self.optimizer.step()

                    total_loss+=loss.item()
                total_loss /= len(batches)
                improvement = (total_loss - losses[-1]) if len(losses) > 0 else 0
                pbar.set_postfix_str(f"Epoch: {epoch}, Batch Loss: {total_loss:.2f}, Improvement: {improvement:.2f}, Min Loss: {min(losses) if len(losses) > 0 else 0}, Batches:{len(batches)}")
                losses.append(total_loss)

                if save_on_best and total_loss > min(losses) and len(losses) % 2 == 0:
                    torch.save(self.model.state_dict(), save_path)
            self.scheduler.step()
        torch.save(self.model.state_dict(), save_path)
        return self.model, losses

class ClimbDDPMGenerator():
    """Moving Climb Generation logic over here to implement automatic conditional feature scaling."""
    def __init__(
            self,
            wall_id: str,
            db_path: str,
            scaler: ClimbsFeatureScaler,
            model: ClimbDDPM,
            model_weights_path: str | None,
            scaler_weights_path: str | None
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = scaler
        self.model = model
        self.timesteps = self.model.timesteps

        if model_weights_path:
            model.load_state_dict(state_dict=clear_compile_keys(model_weights_path),strict=True)
        if scaler_weights_path:
            self.scaler.load_weights(scaler_weights_path)

        with sqlite3.connect(db_path) as conn:
            holds = pd.read_sql_query("SELECT hold_index, x, y, pull_x, pull_y, useability, is_foot, wall_id FROM holds WHERE wall_id = ?",conn,params=(wall_id,))
            scaled_holds = self.scaler.transform_hold_features(holds, to_df=True)
            self.holds_manifold = torch.tensor(scaled_holds[['x','y','pull_x','pull_y']].values, dtype=torch.float32)
            self.holds_lookup = scaled_holds['hold_index'].values
        
        self.holds_lookup = np.concatenate([self.holds_lookup, np.array([-1, -1, -1, -1])])
        
        self.holds_manifold = torch.cat([
            self.holds_manifold,
            torch.tensor(
                [[-2.0, 0.0, -2.0, 0.0],
                [2.0, 0.0, -2.0, 0.0],
                [-2.0, 0.0, 2.0, 0.0],
                [2.0, 0.0, 2.0, 0.0]],dtype=torch.float32)
            ],dim=0)

        self.grade_to_diff = {
            'font': {
                '4a': 10,
                '4b': 11,
                '4c': 12,
                '5a': 13,
                '5b': 14,
                '5c': 15,
                '6a': 16,
                '6a+': 17,
                '6b': 18,
                '6b+': 19,
                '6c': 20,
                '6c+': 21,
                '7a': 22,
                '7a+': 23,
                '7b': 24,
                '7b+': 25,
                '7c': 26,
                '7c+': 27,
                '8a': 28,
                '8a+': 29,
                '8b': 30,
                '8b+': 31,
                '8c': 32,
                '8c+': 33
            }, 
            'v_grade': {
                'V0-': 10,
                'V0': 11,
                'V0+': 12,
                'V1': 13,
                'V1+': 14,
                'V2': 15,
                'V3': 16,
                'V3+': 17,
                'V4': 18,
                'V4+': 19,
                'V5': 20,
                'V5+': 21,
                'V6': 22,
                'V6+': 22.5,
                'V7': 23,
                'V7+': 23.5,
                'V8': 24,
                'V8+': 25,
                'V9': 26,
                'V9+': 26.5,
                'V10': 27,
                'V10+': 27.5,
                'V11': 28,
                'V11+': 28.5,
                'V12': 29,
                'V12+': 29.5,
                'V13': 30,
                'V13+': 30.5,
                'V14': 31,
                'V14+': 31.5,
                'V15': 32,
                'V15+': 32.5,
                'V16': 33
            }
        }
    
    def _build_cond_tensor(self, n, grade, diff_scale, angle):
        diff = self.grade_to_diff[diff_scale][grade]
        df_cond = pd.DataFrame({
            "grade": [diff]*n,
            "quality": [2.9]*n,
            "ascents": [100]*n,
            "angle": [angle]*n
        })

        cond = self.scaler.transform_climb_features(df_cond).T
        return torch.tensor(cond, device=self.device, dtype=torch.float32)
    
    def _project_onto_manifold(self, gen_climbs: Tensor, return_indices=False)-> Tensor:
        """
            Project each generated hold to its nearest neighbor on the hold manifold.
            
            Args:
                gen_climbs: (B, S, H) predicted clean holds
                return_indices: (boolean) Whether to return the hold indices or hold feature coordinates
            Returns:
                projected: (B, S, H) each hold snapped to nearest manifold point
        """
        B, S, H = gen_climbs.shape
        if return_indices:
            climbs = []
            for gen_climb in gen_climbs:
                flat_climb = gen_climb.reshape(-1,H)
                dists = torch.cdist(flat_climb, self.holds_manifold)
                idx = dists.argmin(dim=1)
                idx = idx.detach().numpy()
                holds = self.holds_lookup[idx]
                climb = list(set(holds[holds > 0].tolist()))
                climbs.append(climb)
            return climbs
        else:
            flat_climbs = gen_climbs.reshape(-1,H)
            dists = torch.cdist(flat_climbs, self.holds_manifold)
            idx = dists.argmin(dim=1)
            return self.holds_manifold[idx].reshape(B, S, -1)
        
    def _projection_strength(self, t: Tensor, t_start_projection: float = 0.5):
        """Calculate the weight to assign to the projected holds based on the timestep."""
        a = (t_start_projection-t)/t_start_projection
        strength = 1 - torch.cos(a*torch.pi/2)
        return torch.where(t > t_start_projection, torch.zeros_like(t), strength)
    
    @torch.no_grad()
    def generate(
        self,
        n: int = 1 ,
        angle: int = 45,
        grade: str = 'V4',
        diff_scale: str = 'v_grade',
        deterministic: bool = False,
        projected: bool = True,
        show_steps: bool = False,
        return_indices = True
    )->Tensor:
        """
        Generate a climb or batch of climbs with the given conditions using the standard DDPM iterative denoising process.
        
        :param n: Number of climbs to generate
        :type n: int
        :param angle: Angle of the wall
        :type angle: int
        :param grade: Desired difficulty (V-grade)
        :type grade: int | None
        :return: A Tensor containing the denoised generated climbs as hold sets.
        :rtype: Tensor
        """
        cond_t = self._build_cond_tensor(n, grade, diff_scale, angle)
        x_t = torch.randn((n, 20, 4), device=self.device)
        noisy = x_t.clone()
        t_tensor = torch.ones((n,1), device=self.device)

        for t in range(0, self.timesteps):
            print('.',end='')

            gen_climbs = self.model(noisy, cond_t, t_tensor)

            if projected:
                alpha_p = self._projection_strength(t_tensor)
                projected_climbs = self._project_onto_manifold(gen_climbs)
                gen_climbs = alpha_p*(projected_climbs) + (1-alpha_p)*(gen_climbs)
            
            t_tensor -= 1.0/self.timesteps
            noisy = self.model.forward_diffusion(gen_climbs, t_tensor, x_t if deterministic else torch.randn_like(x_t))

        if projected:
            return self._project_onto_manifold(gen_climbs, return_indices=return_indices)
        else:
            return gen_climbs

#-----------------------------------------------------------------------
# HOLD ROLE CLASSIFICATION
#-----------------------------------------------------------------------
class HoldClassifier(nn.Module):
    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.classifier = classifier
    
    def loss(self, pred_roles: Tensor, true_roles: Tensor):
        """Get the loss from the model's predictions, via cross-entropy loss."""
        return F.cross_entropy(pred_roles, true_roles)
    
    def forward(self, holds: Tensor, cond: Tensor)-> Tensor:
        """Run the forward pass. Predicts the roles for a given (possibly batched) set of holds, given (possibly batched) wall conditions."""
    
    def visualize(self, holdset: Tensor):
        """
        Plot a visualization of the colored holdset.

        :param holdset: Input holdset
        :type holdset: Tensor
        """


class HCTrainer():
    def __init__(self, model: nn.Module, dataset: TensorDataset | None = None, default_batch_size: int = 64):
        self.model = model
        self.dataset = dataset
        self.default_batch_size = default_batch_size
    
    def train(
        self,
        epochs: int,
        save_path: str,
        batch_size: int | None = None
    )-> tuple[nn.Module, np.array]:
        """
        Train the hold classifier.
        
        :param epochs: Number of training epochs.
        :type epochs: int
        :param save_path: Save path
        :type save_path: str
        :param batch_size: Batch size.
        :type batch_size: int | None
        :return: Tuple of (Best Model, training_history)
        :rtype: tuple[Module, Any]
        """

#-----------------------------------------------------------------------
# UTILITY FUNCTIONS
#-----------------------------------------------------------------------
def clear_compile_keys(filepath: str, map_loc: str = "cpu")->dict:
    state_dict = torch.load(filepath, map_location=map_loc)
    new_state_dict = {}
    compile_prefix = "_orig_mod."
    for k, v in state_dict.items():
        if k.startswith(compile_prefix):
            new_k = k.replace(compile_prefix,"")
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def plot_climb(climb_data, title="Generated Climb"):
    """
    Visualizes a climb generated by a neural network.
    
    Args:
        climb_data (np.array): Shape [20, 4]. 
                               Features: [x, y, pull_x, pull_y, role_emb]
        title (str): Title for the plot.
    """
    # role = np.argmax(climb_data[:,5:], axis=1)
    real_holds = climb_data[climb_data[:,0] > -1.1]
    x, y, pull_x, pull_y = real_holds.T

    # 2. Setup the Plot
    # Climbing walls are vertical, so we use a tall figsize
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # 3. Plot Hand Holds (Circles, Blue)
    # We filter using the inverted boolean mask
    ax.scatter(x, y, c='blue', s=90)

    # 5. Plot Pull Vectors (Arrows)
    # quiver(x, y, u, v) plots arrows at (x,y) with direction (u,v)
    ax.quiver(x, y, pull_x, pull_y, 
              color='green', alpha=0.6, 
              angles='xy', scale_units='xy', scale=1, 
              width=0.005, headwidth=4,
              label='Pull Direction', zorder=1)

    # 6. Formatting
    ax.set_title(title)
    ax.set_xlabel("X Position (Normalized)")
    ax.set_ylabel("Y Position (Normalized)")
    ax.set_ylim(-1,1)
    ax.set_xlim(-1,1)
    
    # Important: set aspect to 'equal' so the wall doesn't look stretched
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')


    plt.show()

#Test single-batch memorization to ensure both model architectures are working properly.
def test_single_batch(model: nn.Module, dataset: TensorDataset, steps: int = 1000, lr = 1e-3, decay=0.0):
    if decay == 0:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(params = model.parameters(), lr=lr, weight_decay=decay)

    loader = DataLoader(dataset=dataset, batch_size=64)
    x, c = next(iter(loader))
    losses = []
    with tqdm(range(steps)) as pbar:
        for epoch in pbar:
            optimizer.zero_grad()
            loss = model.loss(x, c)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            if epoch % 10 == 0:
                pbar.set_postfix_str(f"Loss: {loss.item():.4f}, Improvement:{losses[-1]-loss.item():.4f}, Grad Norm:{grad_norm:.4f}, Min loss:{min(losses) if len(losses) > 0 else 0:.5f}")
    print(f"Min loss:{min(losses):.5f}")
    return losses

def moving_average(values, window, gaussian = False):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    #We create the vector to multiply each value by to get the moving average. Essentially a vector of length n
    # in which each weight is 1/n.
    kernel = np.repeat(1.0, window) / window
    if (gaussian == True) :
        if window % 2 == 0:
            window+=1
        x = np.arange(-(window // 2), window // 2 + 1)
        kernel = np.exp(-(x ** 2) / (2 * window ** 2))
        kernel = kernel / np.sum(kernel)
    
    #The convolve function iteratively multiplies the first n values in the values array by the weights array.
    # with the given weights array, it essentially takes the moving average of each N values in the values array.
    return np.convolve(values, kernel, "valid")