"""
DDPM model classes and generator for climb generation.

Ported from model-training/equivariant_projected_diffusion/.
Contains:
- Noiser (U-Net style denoiser)
- ClimbDDPM (diffusion wrapper)
- ClimbsFeatureScaler (data normalization)
- ClimbDDPMGenerator (generation with manifold projection)
"""
import math
import numpy as np
import pandas as pd
import sqlite3
import joblib
import os
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch import Tensor
from sklearn.preprocessing import MinMaxScaler
from app.config import settings

from pathlib import Path

#-----------------------------------------------------------------------
# UNET Diffusion Building Blocks
#-----------------------------------------------------------------------
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
    def __init__(self, hidden_dim=256, layers = 3, in_feature_dim = 12, out_feature_dim = 12, cond_dim = 4, sinusoidal = True):
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

# ---------------------------------------------------------------------------
# Feature scaler
# ---------------------------------------------------------------------------
STYLE_TAGS = ["pinch", "flat"]
SCALED_FEATURES = ['x', 'y', 'pull_x', 'pull_y']
BINARY_FEATURES = ['is_foot'] + STYLE_TAGS
HOLD_FEATURE_COLS = SCALED_FEATURES + BINARY_FEATURES
NUM_HOLD_FEATURES = len(HOLD_FEATURE_COLS)
NUM_ROLES = 5
COND_FEATURES = ['grade', 'quality', 'ascents', 'angle']

class ClimbsFeatureScaler:
    def __init__(self, weights_path: str | None = None):
        self.set_weights = False
        self.cond_features_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.hold_position_scaler = MinMaxScaler(feature_range=(-1, 1))
        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)

    def save_weights(self, path: str):
        state = {
            'cond_scaler': self.cond_features_scaler,
            'hold_position_scaler': self.hold_position_scaler,
        }
        joblib.dump(state, path)
    
    def load_weights(self, path: str):
        state = joblib.load(path)
        self.cond_features_scaler = state['cond_scaler']
        self.hold_position_scaler = state['hold_position_scaler']
        self.set_weights = True
        
    def transform(self, climbs_to_fit: pd.DataFrame, holds_to_fit: pd.DataFrame):
        """
        Fit/transform the scalers on climbs and holds dataframes.
        
        Climbs: log-transform quality/ascents, then MinMaxScale [grade, quality, ascents, angle] to [-1,1].
        Holds:  MinMaxScale [x, y, pull_x, pull_y] to [-1,1].
                Multiply Pull_y and Pull_x by useability.
                Parse style tags from JSON → binary columns.
        """
        # --- Climbs ---
        scaled_climbs = climbs_to_fit.copy()
        scaled_climbs = self._apply_log_transforms(scaled_climbs)
        
        if self.set_weights:
            scaled_climbs[COND_FEATURES] = self.cond_features_scaler.transform(scaled_climbs[COND_FEATURES])
        else:
            scaled_climbs[COND_FEATURES] = self.cond_features_scaler.fit_transform(scaled_climbs[COND_FEATURES])

        # --- Holds ---
        scaled_holds = holds_to_fit.copy()
        scaled_holds = self._apply_hold_transforms(scaled_holds)
        
        if self.set_weights:
            scaled_holds[SCALED_FEATURES] = self.hold_position_scaler.transform(scaled_holds[SCALED_FEATURES])
        else:
            scaled_holds[SCALED_FEATURES] = self.hold_position_scaler.fit_transform(scaled_holds[SCALED_FEATURES])
            self.set_weights = True
        
        return scaled_climbs, scaled_holds
    
    def _apply_log_transforms(self, dfc: pd.DataFrame) -> pd.DataFrame:
        """Log-transform quality and ascents for better scaling."""
        dfc['quality'] -= 3
        dfc['quality'] = np.log(1 - dfc['quality'])
        dfc['ascents'] = np.log(dfc['ascents'])
        return dfc
    
    def _apply_hold_transforms(self, dfh: pd.DataFrame) -> pd.DataFrame:
        """
        Parse style tags from JSON, create binary columns, and map all binary
        features (is_foot + style tags) from {0, 1} → {-1, 1}.
        No multiplication into pull vectors — pull_x/pull_y stay raw before scaling.
        """
        # Parse JSON tags into multi-hot columns
        for tag in STYLE_TAGS:
            dfh[tag] = dfh['tags'].apply(
                lambda t: 1.0 if isinstance(t, str) and tag in json.loads(t) else 0.0
            )

        dfh['pull_x'] *= dfh['useability']
        dfh['pull_y'] *= dfh['useability']
        
        for col in BINARY_FEATURES:
            dfh[col] = dfh[col].astype(float)
        
        dfh = dfh.drop(columns=['useability','tags'])
        
        return dfh
    
    def transform_climb_features(self, climbs_to_transform: pd.DataFrame, to_df: bool = False):
        """Turn conditional climb features into normalized features for inference."""
        dfc = climbs_to_transform.copy()
        dfc = self._apply_log_transforms(dfc)
        if to_df:
            dfc[COND_FEATURES] = self.cond_features_scaler.transform(dfc[COND_FEATURES])
        else:
            dfc = self.cond_features_scaler.transform(dfc[COND_FEATURES])
            dfc = dfc.T
        return dfc
    
    def transform_hold_features(self, holds_to_transform: pd.DataFrame, to_df: bool = False):
        """Turn hold features into normalized features for inference.
        
        Input DataFrame should have columns: x, y, pull_x, pull_y, useability, is_foot, tags, layout_id
        Returns all 11 hold feature columns (or their transposed array).
        """
        dfh = holds_to_transform.copy()
        dfh = self._apply_hold_transforms(dfh)
        dfh[SCALED_FEATURES] = self.hold_position_scaler.transform(dfh[SCALED_FEATURES])
        
        if to_df:
            return dfh
        else:
            return dfh[HOLD_FEATURE_COLS].values.T



# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def clear_compile_keys(filepath: Path | str, map_loc: str = "cpu") -> dict:
    """Strip torch.compile prefixes from state dict keys."""
    state_dict = torch.load(filepath, map_location=map_loc)
    new_state_dict = {}
    prefix = "_orig_mod."
    for k, v in state_dict.items():
        new_state_dict[k.removeprefix(prefix)] = v
    return new_state_dict

def zero_com(climbs: Tensor, dim: int)->Tensor:
    """Perform Zero-Center-Of-Mass transformation on a batched input climbs Tensor of shape [B,S,dim], allowing the model being trained to be translation-invariant."""
    new = climbs.clone()
    com = new[:,:,:dim].mean(dim=1, keepdim=True)
    new[:,:,:dim] -= com
    return new

# ---------------------------------------------------------------------------
# Grade lookup table
# ---------------------------------------------------------------------------

GRADE_TO_DIFF = {
    "font": {
        "4a": 10, "4b": 11, "4c": 12,
        "5a": 13, "5b": 14, "5c": 15,
        "6a": 16, "6a+": 17, "6b": 18, "6b+": 19,
        "6c": 20, "6c+": 21,
        "7a": 22, "7a+": 23, "7b": 24, "7b+": 25,
        "7c": 26, "7c+": 27,
        "8a": 28, "8a+": 29, "8b": 30, "8b+": 31,
        "8c": 32, "8c+": 33,
    },
    "v_grade": {
        "V0-": 10, "V0": 11, "V0+": 12,
        "V1": 13, "V1+": 14, "V2": 15,
        "V3": 16, "V3+": 17, "V4": 18, "V4+": 19,
        "V5": 20, "V5+": 21, "V6": 22, "V6+": 22.5,
        "V7": 23, "V7+": 23.5, "V8": 24, "V8+": 25,
        "V9": 26, "V9+": 26.5, "V10": 27, "V10+": 27.5,
        "V11": 28, "V11+": 28.5, "V12": 29, "V12+": 29.5,
        "V13": 30, "V13+": 30.5, "V14": 31, "V14+": 31.5,
        "V15": 32, "V15+": 32.5, "V16": 33,
    },
}

# ---------------------------------------------------------------------------
# ClimbDDPMGenerator
# ---------------------------------------------------------------------------
FEATURE_WEIGHTS = [1.0,1.0,1.0,1.0,0.5,0.5,0.5]

class ClimbDDPMGenerator():
    def __init__(
            self,
            scaler: ClimbsFeatureScaler,
            ddpm: ClimbDDPM,
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = scaler
        self.ddpm = ddpm
        self._cond_cache = {}
        self.holds_manifolds = {}
        self.holds_lookup = {}
        self.deterministic_noise_generator = torch.Generator(device=self.device)
        self.feature_weights = torch.tensor(FEATURE_WEIGHTS)

        try:
            with sqlite3.connect(settings.DB_PATH) as conn:
                holds = pd.read_sql_query("SELECT hold_index, x, y, pull_x, pull_y, useability, is_foot, tags, layout_id FROM holds", conn)
                layout_ids = list(set(holds['layout_id'].values))
            
            scaled_holds = self.scaler.transform_hold_features(holds, to_df=True)
            
            for layout_id in layout_ids:
                df = scaled_holds[scaled_holds['layout_id']==layout_id]
                self.holds_manifolds[layout_id] = torch.tensor(df[['x','y','pull_x','pull_y']].values, dtype=torch.float32)
                self.holds_lookup[layout_id] = df['hold_index'].values
        except Exception as e:
            pass
    
    def log_hold_means(self, layout_id: str | None = None):
        """Log the hold means for each wall."""
        for k, manifold in self.holds_manifolds.items():
            if layout_id == None or layout_id == k:
                means = torch.mean(manifold, dim=0)
                print(f"Wall-id--{k}; Means-- x:{means[0].item()}, y:{means[1].item()}, Px:{means[2].item()}, Py:{means[3].item()} ")

    def _build_cond_tensor(self, n, diff, angle):
        cache_key = (diff, angle)
        if cache_key not in self._cond_cache:
            row = np.array([[diff, 3.0, 1000, float(angle)]])
            scaled = self.scaler.transform_climb_features(pd.DataFrame(row, columns=['grade','quality','ascents','angle']))
            self._cond_cache[cache_key] = scaled[0]
        base = self._cond_cache[cache_key]
        tiled = np.tile(base, (n,1))
        return torch.tensor(tiled, device=self.device, dtype=torch.float32)
    
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
        dists = torch.cdist(flat_climbs[:,:,:NUM_FEATURES]*self.feature_weights, offset_manifold*self.feature_weights)

        idx = dists.argmin(dim=1)
        return torch.cat([offset_manifold[idx].reshape(B, S, -1),gen_climbs[:,:,NUM_FEATURES:]],dim=2)
    
    def _get_offset_manifold(self, layout_id: str, x_offset: float | None)-> Tensor:
        """Method for offsetting the current holds-manifold such that mean-x and mean-y is 0"""
        offset_manifold = self.holds_manifolds[layout_id].clone()
        means = torch.mean(offset_manifold, dim=0).round(decimals=3)
        if x_offset is None:
            x_offset = -(means[0].item())
        y_offset = -(means[1].item())
        
        offset_manifold[:,0] += x_offset
        offset_manifold[:,1] += y_offset
        means = torch.mean(offset_manifold, dim=0)

        return offset_manifold

    def _find_optimal_translation(
        self,
        gen_climbs: Tensor,           # (B, S, H)
        offset_manifold: Tensor,      # (M, H)
        x_offsets: Tensor,            # (Nx,)
        y_offsets: Tensor,            # (Ny,)
    ) -> tuple[Tensor, Tensor]:
        """
        Find the (dx, dy) translation per batch item that minimises total
        nearest-neighbour projection distance onto the hold manifold.

        Returns:
            best_dx: (B,)  optimal x translation for each climb
            best_dy: (B,)  optimal y translation for each climb
        """
        B, S, H = gen_climbs.shape
        
        # Create a null mask so that the null-hold positions don't contribute to the distance metric.
        null_mask = (gen_climbs[:,:,-1] < 0.95 ).float()
        Nx = x_offsets.shape[0]
        Ny = y_offsets.shape[0]

        # Build a (Nx*Ny, H) manifold shift table — only x/y cols (0 and 2) move
        # Shape: (Nx, Ny, H)
        shifts = torch.zeros(Nx, Ny, H, device=gen_climbs.device)
        shifts[:, :, 0] = x_offsets.unsqueeze(1)   # broadcast over Ny
        shifts[:, :, 1] = y_offsets.unsqueeze(0)   # broadcast over Nx
        G = Nx * Ny
        shifts = shifts.reshape(G, H)         # (G, H)  G = grid size

        # Translate climbs: (B, S, H) + (G, H) -> (G, B, S, H)
        translated = gen_climbs.unsqueeze(0) + shifts.unsqueeze(1).unsqueeze(2)

        # Flatten holds dim for cdist: (G*B, S, H)
        flat = translated.reshape(G * B, S, H)

        # Nearest-neighbour distances to manifold: (G*B, S, M) -> (G*B, S)
        dists = torch.cdist(flat[:,:,:(NUM_FEATURES)]*self.feature_weights, offset_manifold*self.feature_weights)              # (G*B, S, M)
        nn_dists = dists.min(dim=2).values                      # (G*B, S)
        batch_dist = nn_dists.reshape(G, B, S)                  # (G, B, S)
        batch_dist = null_mask.unsqueeze(0) * batch_dist

        total_dist = batch_dist.sum(dim=2)                      # (G, B)

        # Best grid point per batch item
        best_g = total_dist.argmin(dim=0)                       # (B,)
        best_dx = x_offsets[best_g // Ny]
        best_dy = y_offsets[best_g % Ny]

        return best_dx, best_dy

    def _project_onto_indices_with_translation(
        self,
        gen_climbs: Tensor,
        offset_manifold: Tensor,
        layout_id: str,
        x_offsets: Tensor | None = None,
        y_offsets: Tensor | None = None,
    ) -> list[list[list[int]]]:

        if x_offsets is None:
            x_offsets = torch.linspace(-0.5, 0.5, 51, device=gen_climbs.device)
        if y_offsets is None:
            y_offsets = torch.linspace(-0.5, 0.5, 51, device=gen_climbs.device)

        best_dx, best_dy = self._find_optimal_translation(
            gen_climbs, offset_manifold, x_offsets, y_offsets
        )
        print(f"Best Dx:{best_dx}, Best Dy: {best_dy}")

        # Apply per-climb optimal translation  (B, S, H)
        B, S, H = gen_climbs.shape
        translation = torch.zeros(B, 1, H, device=gen_climbs.device)
        translation[:, 0, 0] = best_dx   # x col
        translation[:, 0, 1] = best_dy   # y col  (pull_x/pull_y cols 1,3 left alone)
        translated_climbs = gen_climbs + translation

        # Now do the standard index projection on the translated climbs
        return self._project_onto_indices(translated_climbs, offset_manifold, layout_id)

    def _project_onto_indices(self, gen_climbs: Tensor, offset_manifold: Tensor, layout_id: str) -> list[list[list[int]]]:
        """Project climb onto the final hold indices (and remove null holds)"""
        B, S, H = gen_climbs.shape

        roles = torch.argmax(gen_climbs[:,:,(NUM_FEATURES):], dim=2).detach().numpy()

        flat_climbs = gen_climbs.reshape(-1,H)                  # (B*S, H)

        dists = torch.cdist(flat_climbs[:,:,:(NUM_FEATURES)]*self.feature_weights, offset_manifold*self.feature_weights)              # (B*S, H, M)
        
        idx = dists.argmin(dim=1)
        holds = self.holds_lookup[layout_id][idx]
        holds = holds.reshape(B, S)
        
        is_null = (holds == -1)
        roles[is_null] = 4
        
        climbs = np.stack([holds, roles], axis=2)
        
        deduped_climbs = []
        for c in climbs:
            valid_mask = c[:, 1] != 4
            c_valid = c[valid_mask]
            c_sorted = c_valid[c_valid[:, 1].argsort()]
            _, unique_indices = np.unique(c_sorted[:, 0], return_index=True)
            deduped_climbs.append(c_sorted[unique_indices].tolist())

        return deduped_climbs
    
    def _projection_strength(self, t: Tensor, t_start_projection: float = 0.8):
        """Calculate the weight to assign to the projected holds based on the timestep."""
        a = (t_start_projection-t)/t_start_projection
        strength = 1 - torch.cos(a*torch.pi/2)
        return torch.where(t > t_start_projection, torch.zeros_like(t), strength).unsqueeze(2)
    
    @torch.no_grad()
    def generate(
        self,
        layout_id: str,
        n: int,
        angle: int,
        grade: str,
        diff_scale: str,
        timesteps: int,
        deterministic: bool,
        t_start_projection: float,
        x_offset: float | None,
        guidance_value: float,
        seed: int,
    )->list[list[list[int]]]:
        """
        Generate a climb or batch of climbs with the given conditions using the standard DDPM iterative denoising process.

        :param layout_id: The layout id on which to generate the climb.
        :type layout_id: str
        :param n: The number of climbs to generate.
        :type n: int
        :param angle: The current wall angle.
        :type angle: int
        :param grade: The desired grade.
        :type grade: str
        :param diff_scale: The desired difficulty scale (V-scale or Font).
        :type diff_scale: str
        :param timesteps: Model setting: Number of diffusion timesteps to run. Higher results in better quality (Should be a divisor of 100 to retain markovian properties)
        :type timesteps: int
        :param deterministic: Whether to use the original noise vector in successive diffusion steps, or use a new noise vector each time.
        :type deterministic: bool
        :param t_start_projection: Point in the generation process to begin the projection steps. Earlier is better but more expensive.
        :type t_start_projection: float
        :param x_offset: Offset the climb on the X-axis.
        :type x_offset: float | None
        :param guidance_value: The guidance value to use for CFG generation.
        :type guidance_value: float
        :param seed: The random integer used to seed deterministic climb generation
        :type seed: int
        :return: A set of generated climbs according to the specified 
        :rtype: list[list[list[int]]]
        """
        # Seed Noise Generator
        if deterministic:
            self.deterministic_noise_generator.manual_seed(seed)
        
        # Handle manifold offset
        auto = True if x_offset is None else False
        offset_manifold = self._get_offset_manifold(layout_id, x_offset)

        # CORE LOGIC
        cond_t = self._build_cond_tensor(n, grade, diff_scale, angle)
        x_t = torch.randn((n, 20, NUM_ROLES+NUM_FEATURES), device=self.device, generator=self.deterministic_noise_generator) if deterministic else torch.randn((n, 20, NUM_ROLES+NUM_FEATURES), device=self.device)
        noisy = x_t.clone()
        t_tensor = torch.ones((n,1), device=self.device)
        
        for _ in range(0, timesteps):
            gen_climbs = self.ddpm(noisy, cond_t, t_tensor)

            if t_tensor[0].item() < t_start_projection:
                alpha_p = self._projection_strength(t_tensor, t_start_projection)
                projected_climbs = self._project_onto_manifold(gen_climbs, offset_manifold)
                gen_climbs = alpha_p*(projected_climbs) + (1-alpha_p)*(gen_climbs)
            
            t_tensor -= 1.0/timesteps
            noisy = self.ddpm.forward_diffusion(gen_climbs, t_tensor, x_t if deterministic else torch.randn_like(x_t))
        
        if auto:
            return self._project_onto_indices_with_translation(gen_climbs, offset_manifold, layout_id)
        return self._project_onto_indices(gen_climbs, offset_manifold, layout_id)

# ---------------------------------------------------------------------------
# Global ClimbGenerator Instance For Dependency Injection
# ---------------------------------------------------------------------------
def reset_generator():
    scaler = ClimbsFeatureScaler(
        weights_path=settings.SCALER_WEIGHTS_PATH
    )
    ddpm = ClimbDDPM(
        model=Noiser(),
        weights_path=settings.DDPM_WEIGHTS_PATH,
    )

    ddpm.eval()

    generator = ClimbDDPMGenerator(
        scaler=scaler,
        ddpm=ddpm
    )

    generator.log_hold_means()
    return generator

generator = reset_generator()