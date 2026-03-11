import math
import torch
from torch import nn, Tensor
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import sqlite3
from torchinfo import summary
from climb_conversion import HOLD_FEATURE_COLS
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd

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
    def __init__(self, hidden_dim=128, layers=3, in_feature_dim=16, out_feature_dim=11, cond_dim=4, sinusoidal=True):
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
        self.null_roles = torch.full((1, 20, 5), -1, dtype=torch.float32, device=self.device)

        self.combine_t_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.init_conv = ResidualBlock1D(in_feature_dim, hidden_dim, hidden_dim)

        self.down_blocks = nn.ModuleList([ResidualBlock1D(hidden_dim*(i+1), hidden_dim*(i+2), hidden_dim) for i in range(layers)])
        self.bottom_block = ResidualBlock1D(hidden_dim*(layers+1), hidden_dim*(layers), hidden_dim)
        self.up_blocks = nn.ModuleList([ResidualBlock1D(hidden_dim*(i)*2, hidden_dim*(i-1), hidden_dim) for i in range(layers,1,-1)])

        self.top_block = ResidualBlock1D(hidden_dim*2,hidden_dim, hidden_dim)

        self.head = nn.Conv1d(hidden_dim, out_feature_dim, 1)
    
    def forward(
        self, 
        climbs: Tensor, 
        roles: Tensor | None, 
        cond: Tensor | None, 
        t: Tensor,
        role_mask: Tensor | None = None,
        cond_mask: Tensor | None = None
    ) -> Tensor:
        
        (B, S, H) = climbs.shape
        emb_t = self.time_mlp(t)

        # --- Roles Handling ---
        if roles is None:
            roles = self.null_roles.expand(B, -1, -1)
        elif role_mask is not None:
            null_r = self.null_roles.expand(B, -1, -1)
            roles = torch.where(role_mask.view(B, 1, 1), null_r, roles)
            
        h = torch.cat([climbs, roles], dim=2)
        
        # --- Cond Handling ---
        if cond is None:
            emb_c = self.null_cond_emb.expand(B, -1)
        else:
            emb_c = self.cond_mlp(cond)
            if cond_mask is not None:
                null_c = self.null_cond_emb.expand(B, -1)
                emb_c = torch.where(cond_mask.view(B, 1), null_c, emb_c)
                
        emb_c = self.combine_t_mlp(torch.cat([emb_t, emb_c], dim=1))
        emb_h = self.init_conv(h.transpose(1,2), emb_c)
        
        residuals = []
        for layer in self.down_blocks:
            residuals.append(emb_h)
            emb_h = layer(emb_h, emb_c)
        
        emb_h = self.bottom_block(emb_h, emb_c)
        
        for layer in self.up_blocks:
            residual = residuals.pop()
            emb_h = layer(torch.cat([emb_h, residual], dim=1), emb_c)
        
        residual = residuals.pop()
        emb_h = self.top_block(torch.cat([emb_h, residual], dim=1), emb_c)
        result = self.head(emb_h).transpose(1,2)

        return result

# ----------------------------------------------------------------------
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
    
    def loss(self, sample_climbs: Tensor, roles: Tensor | None, cond: Tensor | None, dropout: tuple[float, float] = (0.0, 0.0)):
        """Perform a diffusion Training step and return the loss."""
        B, S, H = sample_climbs.shape
        t = torch.round(torch.rand(B, 1, device=self.device), decimals=2)
        noise = torch.randn((B, S, H), device=self.device)
        noisy = self.forward_diffusion(sample_climbs, t, noise)
        
        # Generate per-sample boolean masks
        role_mask = (torch.rand(B, device=self.device) < dropout[0]) if (roles is not None and dropout[0] > 0) else None
        cond_mask = (torch.rand(B, device=self.device) < dropout[1]) if (cond is not None and dropout[1] > 0) else None
        
        # Forward pass with masks
        pred_noise = self.model(noisy, roles, cond, t, role_mask=role_mask, cond_mask=cond_mask)
        
        if roles is not None:
            is_real = (roles[:,:,4] == 0).float().unsqueeze(2)
            return F.mse_loss(pred_noise * is_real, noise * is_real)
        return F.mse_loss(pred_noise, noise)
    
    def predict_clean(self, noisy, roles, cond, t):
        """Return predicted clean data."""
        a = self._cos_alpha_bar(t)
        prediction = self.model(noisy, roles, cond, t)
        clean = (noisy - torch.sqrt(1-a)*prediction)/torch.sqrt(a)
        return clean
    
    def predict_cfg(self, noisy, roles, cond, t, guidance_value=1.0):
        a = self._cos_alpha_bar(t)
        cf_pred = self.model(noisy, None, None, t)
        pred = self.model(noisy, roles, cond, t)
        cfg = cf_pred+(pred-cf_pred)*guidance_value
        clean = (noisy - torch.sqrt(1-a)*cfg)/torch.sqrt(a)
        return clean
    
    def forward_diffusion(self, clean: Tensor, t: Tensor, noise: Tensor)-> Tensor:
        """Perform forward diffusion to add noise to clean data based on noise adding schedule."""
        a = self._cos_alpha_bar(t)
        return torch.sqrt(a) * clean + torch.sqrt(1-a) * noise
    
    def forward(self, noisy, roles, cond, t):
        return self.predict_clean(noisy, roles, cond, t)

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
        dropout: tuple[float,float] = (0.25, 0.25)
    ) -> tuple[nn.Module, list]:
        
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
        print(f"Beginning Training. {epochs} Epochs. {len(batches)} Batches. Model Params: {sum([p.numel() for p in self.model.parameters()])}")

        with tqdm(range(epochs)) as pbar:
            pbar.set_postfix_str(f"Epoch: {0}, Batches:{len(batches)}")
            for epoch in pbar:
                total_loss = 0
                for x, r, c in batches:
                    x, r, c = x.to(self.device), r.to(self.device), c.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    # Pass dropout cleanly into the loss function
                    loss = self.model.loss(x, r, c, dropout=dropout)
                    
                    if clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    
                total_loss /= len(batches)
                improvement = (total_loss - losses[-1]) if len(losses) > 0 else 0
                min_loss = min(losses) if len(losses) > 0 else float('inf')
                
                pbar.set_postfix_str(f"Epoch: {epoch}, Batch Loss: {total_loss:.2f}, Improvement: {improvement:.2f}, Min Loss: {min_loss if min_loss != float('inf') else 0:.2f}, Batches:{len(batches)}")
                
                # FIXED: Save when total_loss is LESS than min(losses)
                if save_on_best and save_path and len(losses) > 0 and total_loss < min_loss and len(losses) % 2 == 0:
                    torch.save(self.model.state_dict(), save_path)
                    
                losses.append(total_loss)
            self.scheduler.step()
            
        if save_path is not None:
            torch.save(self.model.state_dict(), save_path)
            
        return self.model, losses
#-----------------------------------------------------------------------
# GENERATION
#-----------------------------------------------------------------------
class ClimbDDPMGenerator():
    # Role indices
    ROLE_START  = 0
    ROLE_FINISH = 1
    ROLE_HAND   = 2
    ROLE_FOOT   = 3
    ROLE_NULL   = 4
    NUM_ROLES   = 5
    SEQ_LEN     = 20
    NUM_FEATURES = 11  # x, y, pull_x, pull_y, useability, is_foot, sloper, pinch, macro, flat, jug

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
            holds = pd.read_sql_query(
                "SELECT hold_index, x, y, pull_x, pull_y, useability, is_foot, tags, layout_id FROM holds WHERE layout_id = ?",
                conn, params=(wall_id,)
            )
            scaled_holds = self.scaler.transform_hold_features(holds, to_df=True)
            self.holds_manifold = torch.tensor(scaled_holds[HOLD_FEATURE_COLS].values, device=self.device, dtype=torch.float32)
            self.holds_lookup = scaled_holds['hold_index'].values

        self.holds_lookup = np.concatenate([self.holds_lookup])

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

    def _build_roles_tensor(self, n: int) -> Tensor:
        """
        Build a [n, SEQ_LEN, NUM_ROLES] one-hot roles tensor for a batch of climbs.
        Ordering per climb: start holds → shuffled hand/foot holds → finish holds → null padding.

        Role counts are sampled uniformly:
            start:  1–2
            hand:   2–8
            foot:   2–4
            finish: 1–2
        """
        roles = torch.zeros((n, self.SEQ_LEN, self.NUM_ROLES), dtype=torch.float32, device=self.device)
        for b in range(n):
            idx = 0
            n_start  = np.random.randint(1, 3)
            n_foot   = np.random.randint(2, 5)
            n_hand   = np.random.randint(2, 9)
            n_finish = np.random.randint(1, 3)

            for _ in range(n_start):
                roles[b, idx, self.ROLE_START] = 1.0
                idx += 1

            middle = [self.ROLE_FOOT] * n_foot + [self.ROLE_HAND] * n_hand
            for role in middle:
                roles[b, idx, role] = 1.0
                idx += 1

            for _ in range(n_finish):
                roles[b, idx, self.ROLE_FINISH] = 1.0
                idx += 1

            # Pad remainder with null role
            roles[b, idx:, self.ROLE_NULL] = 1.0

        return roles

    def _project_onto_manifold(self, gen_climbs: Tensor, offset_manifold: Tensor) -> Tensor:
        """
        Project each generated hold to its nearest neighbor on the hold manifold using
        all 11 feature dimensions for distance. Returns full 11-dim feature vectors.

        Args:
            gen_climbs: (B, S, 11) predicted clean holds
            offset_manifold: (M, 11) manifold with x-coordinate offset applied
        Returns:
            projected: (B, S, 11) each hold snapped to nearest manifold point
        """
        B, S, H = gen_climbs.shape
        flat_climbs = gen_climbs.reshape(-1, H)
        dists = torch.cdist(flat_climbs, offset_manifold)
        idx = dists.argmin(dim=1)
        return self.holds_manifold[idx].reshape(B, S, -1)

    def _project_onto_indices(self, gen_climbs: Tensor, offset_manifold: Tensor) -> list[list[int]]:
        """Final projection: snap to nearest manifold point and return hold indices (null holds excluded)."""
        B, S, H = gen_climbs.shape
        climbs = []
        for gen_climb in gen_climbs:
            flat_climb = gen_climb.reshape(-1, H)
            dists = torch.cdist(flat_climb, offset_manifold)
            idx = dists.argmin(dim=1).cpu().numpy()
            holds = self.holds_lookup[idx]
            climb = list(set(holds[holds > 0].tolist()))
            climbs.append(climb)
        return climbs

    def _projection_strength(self, t: Tensor, t_start_projection: float = 0.8):
        """Calculate the weight to assign to the projected holds based on the timestep."""
        a = (t_start_projection - t) / t_start_projection
        strength = 1 - torch.cos(a * torch.pi / 2)
        return torch.where(t > t_start_projection, torch.zeros_like(t), strength)

    @torch.no_grad()
    def generate(
        self,
        n: int = 1,
        angle: int = 45,
        grade: str = 'V4',
        diff_scale: str = 'v_grade',
        guidance_value: float = 3.0,
        deterministic: bool = False
    ) -> list[list[int]]:
        """
        Generate a batch of climbs using CFG-guided DDPM iterative denoising.

        :param n: Number of climbs to generate
        :param angle: Wall angle in degrees
        :param grade: Desired difficulty (e.g. 'V4')
        :param diff_scale: Grade scale ('v_grade' or 'font')
        :param guidance_value: CFG guidance scale (higher = stronger conditioning)
        :param deterministic: If True, reuse the initial noise tensor during renoising
        :return: List of hold-index lists, one per generated climb
        """
        cond_t  = self._build_cond_tensor(n, grade, diff_scale, angle)
        roles_t = self._build_roles_tensor(n)
        x_t     = torch.randn((n, self.SEQ_LEN, self.NUM_FEATURES), device=self.device)
        noisy   = x_t.clone()
        t_tensor = torch.ones((n, 1), device=self.device)

        # Randomly offset the manifold in x so climbs aren't always wall-centred
        x_offset = np.random.randn()
        offset_manifold = self.holds_manifold.clone()
        offset_manifold[:, 0] += x_offset * 0.1

        for _ in range(self.timesteps):
            print('.', end='')

            gen_climbs = self.model.predict_cfg(noisy, roles_t, cond_t, t_tensor, guidance_value)

            alpha_p = self._projection_strength(t_tensor)
            projected_climbs = self._project_onto_manifold(gen_climbs, offset_manifold)
            gen_climbs = alpha_p * projected_climbs + (1 - alpha_p) * gen_climbs

            t_tensor -= 1.0 / self.timesteps
            noise = x_t if deterministic else torch.randn_like(x_t)
            noisy = self.model.forward_diffusion(gen_climbs, t_tensor, noise)

        return self._project_onto_indices(gen_climbs, offset_manifold)
