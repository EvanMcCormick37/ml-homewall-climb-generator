import math
import torch
from torch import nn, Tensor
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import sqlite3
from torchinfo import summary
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from climb_conversion import ClimbsFeatureScaler
from diffusion_utils import (
    DDPM_WEIGHTS_PATH,
    SCALER_WEIGHTS_PATH,
    GRADE_TO_DIFF,
    HC_WEIGHTS_PATH,
    DB_PATH,
    SinusoidalPositionEmbeddings,
    clear_compile_keys,
    plot_climb,
    zero_com,
    test_single_batch,
    moving_average
)

#-----------------------------------------------------------------------
# UNET Diffusion Building Blocks
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
# UNET Diffusion Building Blocks
#-----------------------------------------------------------------------
from numpy import save


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
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.shortcut(x)


class Noiser(nn.Module):
    """Noiser class with concatenation U-Net architecture, learnable null embeddings, and zero-COM input projection."""
    def __init__(self, hidden_dim=128, layers = 3, in_feature_dim = 16, out_feature_dim = 11, cond_dim = 4, sinusoidal = True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.null_cond_emb = nn.Parameter(torch.randn(1, hidden_dim, device=self.device))

        self.combine_t_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.init_conv = ResidualBlock1D(in_feature_dim, hidden_dim, hidden_dim)

        self.down_blocks = nn.ModuleList([ResidualBlock1D(hidden_dim*(i+1), hidden_dim*(i+2), hidden_dim) for i in range(layers)])
        # No concatenation of inputs for bottom block
        self.bottom_block = ResidualBlock1D(hidden_dim*(layers+1), hidden_dim*(layers), hidden_dim)
        # Concatenate inputs for up blocks
        self.up_blocks = nn.ModuleList([ResidualBlock1D(hidden_dim*(i)*2, hidden_dim*(i-1), hidden_dim) for i in range(layers,1,-1)])

        self.top_block = ResidualBlock1D(hidden_dim*2,hidden_dim, hidden_dim)

        self.head = nn.Conv1d(hidden_dim, out_feature_dim, 1)
    
    def forward(self, climbs: Tensor, cond: Tensor | None, t: Tensor)-> Tensor:
        """
        Run denoising pass. Predicts the added noise from the noisy data.
        
        :param climbs: Tensor with hold-set features, including conditional features and hold roles. [B, S, H]
        :param cond: Tensor with climb conditional variables. [B, 4]
        :param t: Tensor with timestep of diffusion. [B, 1]
        :returns: Tensor, the predicted noise added after timestep t.
        """
        (B, S, H) = climbs.shape
        emb_t = self.time_mlp(t)
        
        if cond is not None: 
            emb_c = self.cond_mlp(cond)
        else:
            emb_c = self.null_cond_emb.repeat(B, 1)
        emb_c = self.combine_t_mlp(torch.cat([emb_t, emb_c], dim=1))
        emb_h = self.init_conv(climbs.transpose(1,2), emb_c)
        
        skip_conns = []
        for layer in self.down_blocks:
            skip_conns.append(emb_h)
            emb_h = layer(emb_h, emb_c)
        
        emb_h = self.bottom_block(emb_h, emb_c)
        
        for layer in self.up_blocks:
            skip_conn = skip_conns.pop()
            emb_h = layer(torch.cat([emb_h, skip_conn], dim=1), emb_c)
        
        skip_conn = skip_conns.pop()
        emb_h = self.top_block(torch.cat([emb_h, skip_conn], dim=1), emb_c)
        result = self.head(emb_h).transpose(1,2)

        return result

#-----------------------------------------------------------------------
# DDPM MODEL
#-----------------------------------------------------------------------
class ClimbDDPM(nn.Module):
    def __init__(self, model: nn.Module, weights_path: Path | str | None = None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        if weights_path:
            self.load_state_dict(clear_compile_keys(weights_path))
    
    def _cos_alpha_bar(self, t: Tensor)-> Tensor:
        t = t.view(-1,1,1)
        epsilon = 0.0004
        return  torch.cos((t+epsilon)/(1+2*epsilon)*torch.pi/2)**2
    
    def _composite_alpha_bar(self, t: Tensor, n_features: int) -> Tensor:
        """Get a composite alpha schedule which preserves the last feature column (the "is_null" flag) until t=0.8.
        This allows the Climb DDPM Generator to perform projection on only non-null holds, beginning at t=0.8"""
        
        a_features = self._cos_alpha_bar(t)
        a_null = self._cos_alpha_bar(torch.clamp((t-0.8) / 0.2, 0.0, 1.0))

        a_full = a_features.expand(-1, -1, n_features-1)
        
        return torch.cat([a_full,a_null],dim=2)
    
    def loss(self, sample_climbs: Tensor, cond: Tensor | None):
        """Perform a diffusion Training step and return the loss resulting from the model in the training run.
        Currently returns tuple (loss, real_hold_loss, null_hold_loss)"""
        B, S, H = sample_climbs.shape
        
        t = torch.round(torch.rand(B, 1, device=self.device), decimals=2)
        noise = torch.randn((B, S, H), device = self.device)
        
        noisy = self.forward_diffusion(sample_climbs, t, noise)
        pred_noise = self.model(noisy, cond, t)

        # Mask loss for is_null noise predictions at t < 0.8 (as we have already completely denoised is_null tokens at this point, so the model can't 'see' noise any more)
        null_loss_mask = (t > 0.8).float()
        
        loss = F.mse_loss(pred_noise, noise, reduction = 'none')
        loss[:,:,-1] *= null_loss_mask
        
        return loss.mean()
    
    def predict_clean(self, noisy, cond, t):
        """Return predicted clean data."""
        a = self._cos_alpha_bar(t)
        prediction = self.model(noisy, cond, t)
        clean = (noisy - torch.sqrt(1-a)*prediction)/torch.sqrt(a)
        return clean
    
    def predict_cfg(self, noisy, cond, t, guidance_value=1.0):
        a = self._cos_alpha_bar(t)
        cf_pred = self.model(noisy, None, t)
        pred = self.model(noisy, cond, t)
        cfg = cf_pred+(pred-cf_pred)*guidance_value
        clean = (noisy - torch.sqrt(1-a)*cfg)/torch.sqrt(a)
        return clean
    
    def forward_diffusion(self, clean: Tensor, t: Tensor, noise: Tensor)-> Tensor:
        """Perform forward diffusion to add noise to clean data based on noise adding schedule."""
        (B, S, H) = clean.shape

        a = self._composite_alpha_bar(t, H)
        return torch.sqrt(a) * clean + torch.sqrt(1-a) * noise
    
    def forward(self, noisy, cond, t):
        return self.predict_clean(noisy, cond, t)

#-----------------------------------------------------------------------
# TRAINING
#-----------------------------------------------------------------------
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
        save_path: str | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        dataset: TensorDataset | None = None,
        save_on_best: bool = False,
        clip_grad_norm: bool = True,
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
        :param dropout: Dropout probability for conditional features vector
        :type dropout: float
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
        print(f"Begining Training. {epochs} Epochs. {len(batches)} Batches. Model Params: {sum([p.numel() for p in self.model.parameters()])}")

        with tqdm(range(epochs)) as pbar:
            pbar.set_postfix_str(f"Epoch: {0}, Batches:{len(batches)}")
            for epoch in pbar:
                total_loss = 0
                for x, c in batches:
                    x, c = x.to(self.device), c.to(self.device)

                    self.optimizer.zero_grad()
                    loss = self.model.loss(x, c) + self.model.loss(x, None)
                    if clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    loss.backward()
                    self.optimizer.step()

                    total_loss+=loss.item()
                total_loss /= len(batches)
                improvement = (total_loss - losses[-1]) if len(losses) > 0 else 0
                pbar.set_postfix_str(f"Epoch: {epoch}, Batch Loss: {total_loss:.2f}, Improvement: {improvement:.2f}, Min Loss: {min(losses) if len(losses) > 0 else 0}, Batches:{len(batches)}")
                losses.append(total_loss)

                if save_on_best and total_loss < min(losses) and len(losses) % 2 == 0:
                    torch.save(self.model.state_dict(), save_path)
            self.scheduler.step()
        if save_path is not None:
            torch.save(self.model.state_dict(), save_path)
        return self.model, losses

#-----------------------------------------------------------------------
# GENERATION
#-----------------------------------------------------------------------
class ClimbDDPMGenerator():
    def __init__(
            self,
            wall_id: str,
            scaler: ClimbsFeatureScaler,
            model: ClimbDDPM
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = scaler
        self.model = model
        self.timesteps = 100

        with sqlite3.connect(DB_PATH) as conn:
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

    def _build_cond_tensor(self, n, grade, diff_scale, angle):
        diff = GRADE_TO_DIFF[diff_scale][grade]
        df_cond = pd.DataFrame({
            "grade": [diff]*n,
            "quality": [2.9]*n,
            "ascents": [100]*n,
            "angle": [angle]*n
        })

        cond = self.scaler.transform_climb_features(df_cond)
        return torch.tensor(cond, device=self.device, dtype=torch.float32)
    
    def _project_onto_manifold(self, gen_climbs: Tensor, offset_manifold: Tensor)-> Tensor:
        """
            Project each generated hold to its nearest neighbor on the hold manifold.
            
            Args:
                gen_climbs: (B, S, H) predicted clean holds
                return_indices: (boolean) Whether to return the hold indices or hold feature coordinates
            Returns:
                projected: (B, S, H) each hold snapped to nearest manifold point
        """
        B, S, H = gen_climbs.shape
        flat_climbs = gen_climbs.reshape(-1,H)
        dists = torch.cdist(flat_climbs, offset_manifold)
        idx = dists.argmin(dim=1)
        return self.holds_manifold[idx].reshape(B, S, -1)
        
    def _project_onto_indices(self, gen_climbs: Tensor, offset_manifold: Tensor) -> list[list[int]]:
        """Project climb onto the final hold indices (and remove null holds)"""
        
        B, S, H = gen_climbs.shape

        climbs = []
        for gen_climb in gen_climbs:
            flat_climb = gen_climb.reshape(-1,H)
            dists = torch.cdist(flat_climb, offset_manifold)
            idx = dists.argmin(dim=1)
            idx = idx.detach().numpy()
            holds = self.holds_lookup[idx]
            climb = list(set(holds[holds > 0].tolist()))
            climbs.append(climb)
        
        return climbs
    
    def _projection_strength(self, t: Tensor, t_start_projection: float = 0.8):
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
        deterministic: bool = False
    )->list[list[int]]:
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
        
        # Randomly offset the holds-manifold to allow for climbs to be generated at different x-coordinates around the wall.
        x_offset = np.random.randn()
        offset_manifold = self.holds_manifold.clone()
        offset_manifold[:,0] += x_offset*0.1

        for t in range(0, self.timesteps):
            print('.',end='')

            gen_climbs = self.model(noisy, cond_t, t_tensor)

            alpha_p = self._projection_strength(t_tensor)
            projected_climbs = self._project_onto_manifold(gen_climbs, offset_manifold)
            gen_climbs = alpha_p*(projected_climbs) + (1-alpha_p)*(gen_climbs)
            
            t_tensor -= 1.0/self.timesteps
            noisy = self.model.forward_diffusion(gen_climbs, t_tensor, x_t if deterministic else torch.randn_like(x_t))
        
        return self._project_onto_indices(gen_climbs, offset_manifold)