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
import random
import numpy as np
import pandas as pd
import sqlite3
import joblib
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sklearn.preprocessing import MinMaxScaler
from app.config import settings

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
    def __init__(self, model: nn.Module, weights_path: Path, timesteps: int = 100):
        super().__init__()
        self.model = model
        self.load_state_dict(clear_compile_keys(weights_path))
        self.timesteps = timesteps

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
            dfc = dfc
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
            dfh = dfh
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


# ---------------------------------------------------------------------------
# ClimbDDPMGenerator
# ---------------------------------------------------------------------------

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

        with sqlite3.connect(settings.DB_PATH) as conn:
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
