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
    def __init__(self, hidden_dim=64, layers = 3, sinusoidal = False):
        super().__init__()

        self.time_mlp = nn.Sequential(
            (SinusoidalPositionEmbeddings(hidden_dim) if sinusoidal else nn.Linear(1,hidden_dim)),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim)
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
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

        self.init_conv = ResidualBlock1D(4, hidden_dim, hidden_dim)

        self.down_blocks = nn.ModuleList([ResidualBlock1D(hidden_dim*(i+1), hidden_dim*(i+2), hidden_dim) for i in range(layers)])
        self.up_blocks = nn.ModuleList([ResidualBlock1D(hidden_dim*(i+1), hidden_dim*(i), hidden_dim) for i in range(layers,0,-1)])

        self.head = nn.Conv1d(hidden_dim, 4, 1)
    
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

class ImprovedNoiser(nn.Module):
    def __init__(self, hidden_dim=256, layers=6, input_channels=4, cond_dim=4):
        super().__init__()
        
        # 1. Time Embeddings
        time_dim = hidden_dim // 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 2. Condition Embeddings (Condition + Time)
        # We project the physical conditions (grade, angle, etc.) and fuse them with time
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 3. Initial Convolution
        # Lifts input (x, y, px, py) to hidden dimension
        self.init_conv = nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1)

        # 4. Down-sampling / Encoding Blocks
        self.downs = nn.ModuleList([])
        for _ in range(layers // 2):
            self.downs.append(ResidualBlock1D(hidden_dim, hidden_dim, hidden_dim))

        # 5. Middle Block with ATTENTION
        # This is where the model understands global structure
        self.mid_block1 = ResidualBlock1D(hidden_dim, hidden_dim, hidden_dim)
        self.attn_block = AttentionBlock1D(hidden_dim, num_heads=8)
        self.mid_block2 = ResidualBlock1D(hidden_dim, hidden_dim, hidden_dim)

        # 6. Up-sampling / Decoding Blocks
        self.ups = nn.ModuleList([])
        for _ in range(layers // 2):
            self.ups.append(ResidualBlock1D(hidden_dim, hidden_dim, hidden_dim))

        # 7. Final Head
        self.final_norm = nn.GroupNorm(8, hidden_dim)
        self.final_act = nn.SiLU()
        self.head = nn.Conv1d(hidden_dim, input_channels, kernel_size=1)

    def forward(self, climbs, cond, t):
        """
        :param climbs: [Batch, Sequence_Length, 4] (x, y, pull_x, pull_y)
        :param cond:   [Batch, 4] (grade, quality, ascents, angle)
        :param t:      [Batch, 1] (timesteps 0.0 to 1.0)
        """
        
        # --- Pre-process Inputs ---
        # 1. Process Time
        t_emb = self.time_mlp(t) # [B, time_dim]
        
        # 2. Process Condition + Time
        # Concatenate condition features with time embeddings
        cond_input = torch.cat([cond, t_emb], dim=1)
        emb_c = self.cond_mlp(cond_input) # [B, hidden_dim] - This is fed to ResBlocks

        # 3. Process Climbs
        # Transpose for Conv1d: [B, S, 4] -> [B, 4, S]
        h = self.init_conv(climbs.transpose(1, 2))

        # --- Forward Pass ---
        
        # Encoder
        for layer in self.downs:
            h = layer(h, emb_c)

        # Bottleneck (Attention)
        h = self.mid_block1(h, emb_c)
        h = self.attn_block(h) # Self-attention happens here
        h = self.mid_block2(h, emb_c)

        # Decoder
        for layer in self.ups:
            h = layer(h, emb_c)

        # --- Output ---
        h = self.final_norm(h)
        h = self.final_act(h)
        output = self.head(h)
        
        # Transpose back: [B, 4, S] -> [B, S, 4]
        return output.transpose(1, 2)

class ClimbDDPM(nn.Module):
    def __init__(self, model, predict_noise = False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.timesteps = 100
        self.pred_noise = predict_noise
    
    def _cos_alpha_bar(self, t: Tensor)-> Tensor:
        t = t.view(-1,1,1)
        epsilon = 0.0004
        return  torch.cos((t+epsilon)/(1+2*epsilon)*torch.pi/2)**2
    
    def loss(self, sample_climbs, cond):
        """Perform a diffusion Training step and return the loss resulting from the model in the training run. Currently returns tuple (loss, real_hold_loss, null_hold_loss)"""
        B, S, H = sample_climbs.shape
        t = torch.round(torch.rand(B, 1, device=self.device), decimals=2)

        noisy = self.forward_diffusion(sample_climbs, t)
        pred_clean = self.predict(noisy, cond, t)
        is_real = (sample_climbs[:,:,3] != -2).float().unsqueeze(-1)
        return F.mse_loss(pred_clean*is_real, sample_climbs*is_real)
    
    def predict(self, noisy, cond, t):
        """Return predicted clean data."""
        a = self._cos_alpha_bar(t)
        prediction = self.model(noisy, cond, t)
        if self.pred_noise:
            clean = (noisy - torch.sqrt(1-a)*prediction)/torch.sqrt(a)
        else:
            clean = prediction
        return clean
    
    def forward_diffusion(self, clean: Tensor, t: Tensor, x_0: Tensor | None = None)-> Tensor:
        """Perform forward diffusion to add noise to clean data based on noise adding schedule."""
        if x_0 is None:
            x_0 = torch.randn_like(clean, device=self.device)
        a = self._cos_alpha_bar(t)
        return torch.sqrt(a) * clean + torch.sqrt(1-a) * x_0
    
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
                
                improvement = (total_loss - losses[-1]) if len(losses) > 0 else 0
                pbar.set_postfix_str(f"Epoch: {epoch}, Loss: {total_loss:.2f}, Improvement: {improvement:.2f}, Min Loss: {min(losses) if len(losses) > 0 else 0}, Batches:{len(batches)}")
                losses.append(total_loss)

                if save_on_best and total_loss > min(losses) and len(losses) % 2 == 0:
                    torch.save(self.model.state_dict(), save_path)
            self.scheduler.step()
        torch.save(self.model.state_dict(), save_path)
        return self.model, losses

class ClimbDDPMGenerator():
    """Moving Climb Generation logic over here to implement automatic conditional feature scaling. Need to implement Projected Diffusion."""
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
            self.hold_lookup = self.scaler.transform_hold_features(holds, to_df=True).to_dict('index')
            self.holds_manifold = torch.tensor(self.scaler.transform_hold_features(holds, to_df=False), dtype=torch.float32)

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

        cond = self.scaler.transform_climb_features(df_cond)
        return torch.tensor(cond, device=self.device, dtype=torch.float32)
    
    def _project_onto_manifold(self, gen_climbs: Tensor, hold_manifold: Tensor)-> Tensor:
        """
            Project each generated hold to its nearest neighbor on the hold manifold.
            
            Args:
                climbs: (B, S, H) predicted clean holds
                hold_manifold: (num_holds, H) all valid hold features + null hold
            Returns:
                projected: (B, S, H) each hold snapped to nearest manifold point
        """
        B, S, H = gen_climbs.shape
        flat_climbs = gen_climbs.reshape(-1,H)
        dists = torch.cdist(flat_climbs, hold_manifold)
        idx = dists.argmin(dim=1)
        projected_holds = hold_manifold[idx]

        return projected_holds.reshape(B, S, H)
    
    def _projection_strength(self, t: Tensor, t_start_projection: float = 0.7):
        """Calculate the weight to assign to the projected holds based on the timestep."""
        a = (t_start_projection-t)/t_start_projection
        strength = 1 - torch.cos(a*torch.pi/2)
        return torch.where(t > t_start_projection, torch.zeros_like(t),strength)
    
    @torch.no_grad()
    def generate(
        self,
        n: int = 1 ,
        angle: int = 45,
        grade: str = 'V4',
        diff_scale: str = 'v_grade',
        deterministic: bool = False,
        projected: bool = True,
        show_steps: bool = False
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
            print(t_tensor,end='')

            gen_climbs = self.model(noisy, cond_t, t_tensor)

            if projected:
                alpha_p = self._projection_strength(t_tensor)
                projected_climbs = self._project_onto_manifold(gen_climbs,self.holds_manifold)
                gen_climbs = alpha_p*(projected_climbs) + 1-alpha_p*(gen_climbs)
            
            t_tensor -= 1.0/self.timesteps
            noisy = self.model.forward_diffusion(gen_climbs, t_tensor, x_t if deterministic else None)
        
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