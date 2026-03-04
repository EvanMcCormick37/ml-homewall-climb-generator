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

import torch
import torch.nn as nn
from torch import Tensor, seed
from sklearn.preprocessing import MinMaxScaler
from app.config import settings
from app.database import get_db

from pathlib import Path

# ---------------------------------------------------------------------------
# Neural network building blocks
# ---------------------------------------------------------------------------

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=padding)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        h = self.conv1(x)
        h = self.norm1(h)
        gamma, beta = self.cond_proj(cond).unsqueeze(-1).chunk(2, dim=1)
        h = h * (1 + gamma) + beta
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + self.shortcut(x)

class ResidualBlock1D_V2(nn.Module):
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

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, 0].unsqueeze(1) * embeddings.unsqueeze(0)
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


# ---------------------------------------------------------------------------
# Noiser (U-Net denoiser)
# ---------------------------------------------------------------------------

class Noiser(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        layers: int = 5,
        feature_dim: int = 4,
        cond_dim: int = 4,
        sinusoidal: bool = True,
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim) if sinusoidal else nn.Linear(1, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.combine_t_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.init_conv = ResidualBlock1D(feature_dim, hidden_dim, hidden_dim)
        self.down_blocks = nn.ModuleList(
            [ResidualBlock1D(hidden_dim * (i + 1), hidden_dim * (i + 2), hidden_dim) for i in range(layers)]
        )
        self.up_blocks = nn.ModuleList(
            [ResidualBlock1D(hidden_dim * (i + 1), hidden_dim * i, hidden_dim) for i in range(layers, 0, -1)]
        )
        self.head = nn.Conv1d(hidden_dim, feature_dim, 1)

    def forward(self, climbs: Tensor, cond: Tensor, t: Tensor) -> Tensor:
        emb_t = self.time_mlp(t)
        emb_c = self.cond_mlp(cond)
        emb_c = self.combine_t_mlp(torch.cat([emb_t, emb_c], dim=1))
        emb_h = self.init_conv(climbs.transpose(1, 2), emb_c)

        residuals = []
        for layer in self.down_blocks:
            residuals.append(emb_h)
            emb_h = layer(emb_h, emb_c)

        for layer in self.up_blocks:
            residual = residuals.pop()
            emb_h = residual + layer(emb_h, emb_c)

        return self.head(emb_h).transpose(1, 2)

# ---------------------------------------------------------------------------
# ClimbDDPM (diffusion process)
# ---------------------------------------------------------------------------

class ClimbDDPM(nn.Module):
    def __init__(self, model: nn.Module, weights_path: Path):
        super().__init__()
        self.model = model
        self.load_state_dict(clear_compile_keys(weights_path))

    def _cos_alpha_bar(self, t: Tensor) -> Tensor:
        t = t.view(-1, 1, 1)
        epsilon = 0.0004
        return torch.cos((t + epsilon) / (1 + 2 * epsilon) * torch.pi / 2) ** 2

    def predict_clean(self, noisy, cond, t):
        a = self._cos_alpha_bar(t)
        prediction = self.model(noisy, cond, t)
        return (noisy - torch.sqrt(1 - a) * prediction) / torch.sqrt(a)

    def forward_diffusion(self, clean: Tensor, t: Tensor, x_0: Tensor) -> Tensor:
        a = self._cos_alpha_bar(t)
        return torch.sqrt(a) * clean + torch.sqrt(1 - a) * x_0

    def forward(self, noisy, cond, t):
        return self.predict_clean(noisy, cond, t)


# ---------------------------------------------------------------------------
# Feature scaler
# ---------------------------------------------------------------------------

class ClimbsFeatureScaler:
    """Handles normalization of climb conditional features and hold features."""

    def __init__(self, weights_path: Path | None = None):
        self.cond_features_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.hold_features_scaler = MinMaxScaler(feature_range=(-1, 1))
        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)

    def save_weights(self, path: str):
        state = {
            "cond_scaler": self.cond_features_scaler,
            "hold_scaler": self.hold_features_scaler,
        }
        joblib.dump(state, path)

    def load_weights(self, path: Path):
        state = joblib.load(path)
        self.cond_features_scaler = state["cond_scaler"]
        self.hold_features_scaler = state["hold_scaler"]

    def _apply_hold_transforms(self, dfh: pd.DataFrame) -> pd.DataFrame:
        dfh = dfh.copy()
        dfh["mult"] = dfh["useability"] / ((3 * dfh["is_foot"]) + 1)
        dfh["pull_x"] *= dfh["mult"]
        dfh["pull_y"] *= dfh["mult"]
        return dfh

    def _apply_log_transforms(self, dfc: pd.DataFrame) -> pd.DataFrame:
        dfc = dfc.copy()
        dfc["quality"] -= 3
        dfc["quality"] = np.log(1 - dfc["quality"])
        dfc["ascents"] = np.log(dfc["ascents"])
        return dfc

    def transform_climb_features(self, climbs_to_transform: pd.DataFrame, to_df: bool = False):
        dfc = climbs_to_transform.copy()
        dfc = self._apply_log_transforms(dfc)
        if to_df:
            dfc[["grade", "quality", "ascents", "angle"]] = self.cond_features_scaler.transform(
                dfc[["grade", "quality", "ascents", "angle"]]
            )
        else:
            dfc = self.cond_features_scaler.transform(dfc[["grade", "quality", "ascents", "angle"]])
        return dfc

    def transform_hold_features(self, holds_to_transform: pd.DataFrame, to_df: bool = False):
        dfh = holds_to_transform.copy()
        dfh = self._apply_hold_transforms(dfh)
        if to_df:
            dfh[["x", "y", "pull_x", "pull_y"]] = self.hold_features_scaler.transform(
                dfh[["x", "y", "pull_x", "pull_y"]]
            )
        else:
            dfh = self.hold_features_scaler.transform(dfh[["x", "y", "pull_x", "pull_y"]])
        return dfh


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def clear_compile_keys(filepath: Path, map_loc: str = "cpu") -> dict:
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
# Grade lookup tables
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


def _get_wall_angle(wall_id: str, default_angle: int = 45) -> int:
    """Get the default wall angle from the database. If there is no default wall angle, return 45."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT angle FROM walls WHERE id = ?", (wall_id,)
        ).fetchone()
    if row and row["angle"] is not None:
        return row["angle"]
    return default_angle


#-----------------------------------------------------------------------
# HOLD ROLE CLASSIFICATION
#-----------------------------------------------------------------------

class UNetHoldClassifierLogits(nn.Module):
    def __init__(
        self,
        in_features_dim: int = 4,
        in_cond_dim: int = 4,
        out_dim: int = 5,
        hidden_dim: int = 128,
        n_layers: int = 3,
        weights_path: Path | str | None = None
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cond_emb = nn.Sequential(
            nn.Linear(in_cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.init_conv = ResidualBlock1D_V2(in_features_dim, hidden_dim, hidden_dim)

        self.down_blocks = nn.ModuleList([ResidualBlock1D_V2(hidden_dim*(i+1), hidden_dim*(i+2), hidden_dim) for i in range(n_layers)])
        self.up_blocks = nn.ModuleList([ResidualBlock1D_V2(hidden_dim*(i+1), hidden_dim*(i), hidden_dim) for i in range(n_layers,0,-1)])

        self.head = nn.Conv1d(hidden_dim, out_dim, 1)

        if weights_path:
            self.load_state_dict(clear_compile_keys(weights_path, map_loc = str(self.device)), strict=True)

    def forward(self, x, cond):

        x = zero_com(x, 2)

        cond_emb = self.cond_emb(cond)
        h_emb = self.init_conv(x.transpose(1,2), cond_emb)

        residuals = []
        for layer in self.down_blocks:
            residuals.append(h_emb)
            h_emb = layer(h_emb, cond_emb)
        
        for layer in self.up_blocks:
            resid = residuals.pop()
            h_emb = resid + layer(h_emb, cond_emb)
        
        h_out = self.head(h_emb).transpose(1,2)

        return h_out

# ---------------------------------------------------------------------------
# ClimbDDPMGenerator
# ---------------------------------------------------------------------------
NULL_HOLD_SENTINELS = torch.tensor(
    [[-2.0, 0.0, -2.0, 0.0],
    [2.0, 0.0, -2.0, 0.0],
    [-2.0, 0.0, 2.0, 0.0],
    [2.0, 0.0, 2.0, 0.0]], dtype=torch.float32)

class ClimbDDPMGenerator():
    def __init__(
            self,
            scaler: ClimbsFeatureScaler,
            ddpm: ClimbDDPM,
            hold_classifier: UNetHoldClassifierLogits,
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = scaler
        self.ddpm = ddpm
        self.hold_classifier = hold_classifier
        self._cond_cache = {}
        self.deterministic_noise_generator = torch.Generator(device=self.device)

        with sqlite3.connect(settings.DB_PATH) as conn:
            holds = pd.read_sql_query("SELECT hold_index, x, y, pull_x, pull_y, useability, is_foot, wall_id FROM holds", conn)
            wall_ids = list(set(holds['wall_id'].values))
            scaled_holds = self.scaler.transform_hold_features(holds, to_df=True)
            self.holds_manifolds = {}
            self.holds_lookup = {}
            for wall_id in wall_ids:
                df = scaled_holds[scaled_holds['wall_id']==wall_id]
                self.holds_manifolds[wall_id] = torch.cat([torch.tensor(df[['x','y','pull_x','pull_y']].values, dtype=torch.float32), NULL_HOLD_SENTINELS], dim=0)
                self.holds_lookup[wall_id] = df['hold_index'].values
                self.holds_lookup[wall_id] = np.concatenate([self.holds_lookup[wall_id], np.array([-1, -1, -1, -1])])

    def _build_cond_tensor(self, n, grade, diff_scale, angle):
        cache_key = (grade, diff_scale, angle)
        if cache_key not in self._cond_cache:
            diff = GRADE_TO_DIFF[diff_scale][grade]
            row = np.array([[diff, 3.0, 1000, float(angle)]])
            row[:, 1] = np.log(1 - (row[:, 1] - 3))     # quality
            row[:, 2] = np.log(row[:, 2])               # ascents
            scaled = self.scaler.cond_features_scaler.transform(row)
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
        dists = torch.cdist(flat_climbs, offset_manifold)
        idx = dists.argmin(dim=1)
        return offset_manifold[idx].reshape(B, S, -1)
    
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
        null_mask = ((gen_climbs[:,:,0] < -1.2) | (gen_climbs[:,:,2] > 1.2 ) | (gen_climbs[:,:,0] > 1.2) | (gen_climbs[:,:,2] < -1.2)).float()
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
        dists = torch.cdist(flat, offset_manifold)              # (G*B, S, M)
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
        cond_t: Tensor,
        offset_manifold: Tensor,
        wall_id: str,
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
        return self._project_onto_indices(translated_climbs, cond_t, offset_manifold, wall_id)

    def _project_onto_indices(self, gen_climbs: Tensor, cond_t: Tensor, offset_manifold: Tensor, wall_id: str) -> list[list[list[int]]]:
        """Project climb onto the final hold indices (and remove null holds)"""
        
        B, S, H = gen_climbs.shape

        roles = torch.argmax(self.hold_classifier(gen_climbs, cond_t), dim=2).detach().numpy()

        flat_climbs = gen_climbs.reshape(-1,H)                  # (B*S, H)
        dists = torch.cdist(flat_climbs, offset_manifold)       # (B*S, H, M)
        idx = dists.argmin(dim=1)
        holds = self.holds_lookup[wall_id][idx]
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
    
    def _projection_strength(self, t: Tensor, t_start_projection: float):
        """Calculate the weight to assign to the projected holds based on the timestep."""
        a = (t_start_projection-t)/t_start_projection
        strength = 1 - torch.cos(a*torch.pi/2)
        return torch.where(t > t_start_projection, torch.zeros_like(t), strength).unsqueeze(2)
    
    @torch.no_grad()
    def generate(
        self,
        wall_id: str,
        n: int,
        angle: int,
        grade: str,
        diff_scale: str,
        timesteps: int,
        deterministic: bool,
        t_start_projection: float,
        x_offset: float | None,
        seed: int,
    )->list[list[list[int]]]:
        """
        Generate a climb or batch of climbs with the given conditions using the standard DDPM iterative denoising process.

        :param wall_id: The Wall-ID on which to generate the climb.
        :type wall_id: str
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
        :return: A set of generated climbs according to the specified 
        :rtype: list[list[list[int]]]
        """
        auto=False
        # Seed Noise Generator
        if deterministic:
            self.deterministic_noise_generator.manual_seed(seed)
        
        # Added x-offset logic for more variety
        if x_offset is None:
            val = torch.randn(1, generator=self.deterministic_noise_generator).item() if deterministic else np.random.randn()
            x_offset = max(min(val, 1.0), -1.0)
            auto=True
        
        offset_manifold = self.holds_manifolds[wall_id].clone()
        offset_manifold[:,0] += x_offset*0.1

        # CORE LOGIC
        cond_t = self._build_cond_tensor(n, grade, diff_scale, angle)
        x_t = torch.randn((n, 20, 4), device=self.device, generator=self.deterministic_noise_generator) if deterministic else torch.randn((n, 20, 4), device=self.device)
        noisy = x_t.clone()
        t_tensor = torch.ones((n,1), device=self.device)
        
        for _ in range(0, timesteps):
            gen_climbs = self.ddpm(noisy, cond_t, t_tensor)

            alpha_p = self._projection_strength(t_tensor, t_start_projection)
            projected_climbs = self._project_onto_manifold(gen_climbs, offset_manifold)
            gen_climbs = alpha_p*(projected_climbs) + (1-alpha_p)*(gen_climbs)
            
            t_tensor -= 1.0/timesteps
            noisy = self.ddpm.forward_diffusion(gen_climbs, t_tensor, x_t if deterministic else torch.randn_like(x_t))
        
        if auto:
            return self._project_onto_indices_with_translation(gen_climbs, cond_t, offset_manifold, wall_id)
        return self._project_onto_indices(gen_climbs, cond_t, offset_manifold, wall_id)

# ---------------------------------------------------------------------------
# Global ClimbGenerator Instance For Dependency Injection
# ---------------------------------------------------------------------------
scaler = ClimbsFeatureScaler(
    weights_path=settings.SCALER_WEIGHTS_PATH
)
ddpm = ClimbDDPM(
    model=Noiser(),
    weights_path=settings.DDPM_WEIGHTS_PATH,
)
hold_classifier = UNetHoldClassifierLogits(
    weights_path=settings.HC_WEIGHTS_PATH
)

ddpm.eval()
hold_classifier.eval()

generator = ClimbDDPMGenerator(
    scaler=scaler,
    ddpm=ddpm,
    hold_classifier=hold_classifier
)