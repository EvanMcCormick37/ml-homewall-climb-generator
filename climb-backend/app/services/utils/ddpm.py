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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sklearn.preprocessing import MinMaxScaler


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
    def __init__(self, model: nn.Module, timesteps: int = 100):
        super().__init__()
        self.model = model
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

    def __init__(self, weights_path: str | None = None):
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

    def load_weights(self, path: str):
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
            dfc = dfc.T
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
            dfh = dfh.T
        return dfh


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def clear_compile_keys(filepath: str, map_loc: str = "cpu") -> dict:
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

class ClimbDDPMGenerator:
    """
    Generates climbs using a pre-trained DDPM with manifold projection.

    Holds are loaded per-wall from the database so a single model
    can generate for any wall.
    """

    def __init__(
        self,
        db_path: str,
        scaler: ClimbsFeatureScaler,
        model: ClimbDDPM,
        model_weights_path: str | None = None,
        scaler_weights_path: str | None = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = scaler
        self.model = model.to(self.device).eval()
        self.timesteps = model.timesteps
        self.db_path = db_path

        if model_weights_path:
            self.model.load_state_dict(
                state_dict=clear_compile_keys(model_weights_path), strict=True
            )
        if scaler_weights_path:
            self.scaler.load_weights(scaler_weights_path)

    def _load_wall_holds(self, wall_id: str) -> tuple[torch.Tensor, np.ndarray]:
        """
        Load and scale holds for a specific wall.

        Returns:
            holds_manifold: Tensor of scaled hold features (N, 4)
            holds_lookup: Array of hold_index values (with null-hold sentinels appended)
        """
        with sqlite3.connect(self.db_path) as conn:
            holds = pd.read_sql_query(
                "SELECT hold_index, x, y, pull_x, pull_y, useability, is_foot, wall_id "
                "FROM holds WHERE wall_id = ?",
                conn,
                params=(wall_id,),
            )

        if holds.empty:
            raise ValueError(f"No holds found for wall {wall_id}")

        scaled_holds = self.scaler.transform_hold_features(holds, to_df=True)
        holds_manifold = torch.tensor(
            scaled_holds[["x", "y", "pull_x", "pull_y"]].values, dtype=torch.float32
        )
        holds_lookup = scaled_holds["hold_index"].values

        # Append null-hold sentinel points
        holds_lookup = np.concatenate([holds_lookup, np.array([-1, -1, -1, -1])])
        holds_manifold = torch.cat(
            [
                holds_manifold,
                torch.tensor(
                    [
                        [-2.0, 0.0, -2.0, 0.0],
                        [2.0, 0.0, -2.0, 0.0],
                        [-2.0, 0.0, 2.0, 0.0],
                        [2.0, 0.0, 2.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
            ],
            dim=0,
        )

        return holds_manifold, holds_lookup

    def _build_cond_tensor(
        self, n: int, grade: str, diff_scale: str, angle: int
    ) -> torch.Tensor:
        diff = GRADE_TO_DIFF[diff_scale][grade]
        df_cond = pd.DataFrame(
            {
                "grade": [diff] * n,
                "quality": [2.9] * n,
                "ascents": [100] * n,
                "angle": [angle] * n,
            }
        )
        cond = self.scaler.transform_climb_features(df_cond).T
        return torch.tensor(cond, device=self.device, dtype=torch.float32)

    def _project_onto_manifold(
        self,
        gen_climbs: Tensor,
        holds_manifold: Tensor,
        holds_lookup: np.ndarray,
        return_indices: bool = False,
    ) -> Tensor | list[list[int]]:
        B, S, H = gen_climbs.shape
        if return_indices:
            climbs = []
            for gen_climb in gen_climbs:
                flat_climb = gen_climb.reshape(-1, H)
                dists = torch.cdist(flat_climb, holds_manifold)
                idx = dists.argmin(dim=1).detach().numpy()
                holds = holds_lookup[idx]
                climb = list(set(int(h) for h in holds if h > 0))
                climbs.append(climb)
            return climbs
        else:
            flat_climbs = gen_climbs.reshape(-1, H)
            dists = torch.cdist(flat_climbs, holds_manifold)
            idx = dists.argmin(dim=1)
            return holds_manifold[idx].reshape(B, S, -1)

    def _projection_strength(self, t: Tensor, t_start_projection: float = 0.5) -> Tensor:
        a = (t_start_projection - t) / t_start_projection
        strength = 1 - torch.cos(a * torch.pi / 2)
        return torch.where(t > t_start_projection, torch.zeros_like(t), strength)

    @torch.no_grad()
    def generate(
        self,
        wall_id: str,
        n: int = 1,
        angle: int = 45,
        grade: str = "V4",
        diff_scale: str = "v_grade",
        deterministic: bool = False,
        projected: bool = True,
    ) -> list[list[int]]:
        """
        Generate climbs for a given wall.

        Args:
            wall_id: Wall to generate for (holds loaded from DB)
            n: Number of climbs to generate
            angle: Wall angle in degrees
            grade: Target difficulty grade string
            diff_scale: Grading system ('v_grade' or 'font')
            deterministic: Use fixed noise for reproducibility
            projected: Apply manifold projection

        Returns:
            List of climbs, each a list of hold indices.
        """
        holds_manifold, holds_lookup = self._load_wall_holds(wall_id)
        cond_t = self._build_cond_tensor(n, grade, diff_scale, angle)

        x_t = torch.randn((n, 20, 4), device=self.device)
        noisy = x_t.clone()
        t_tensor = torch.ones((n, 1), device=self.device)

        for _ in range(self.timesteps):
            gen_climbs = self.model(noisy, cond_t, t_tensor)

            if projected:
                alpha_p = self._projection_strength(t_tensor)
                projected_climbs = self._project_onto_manifold(
                    gen_climbs, holds_manifold, holds_lookup
                )
                gen_climbs = alpha_p * projected_climbs + (1 - alpha_p) * gen_climbs

            t_tensor -= 1.0 / self.timesteps
            noisy = self.model.forward_diffusion(
                gen_climbs,
                t_tensor,
                x_t if deterministic else torch.randn_like(x_t),
            )

        if projected:
            return self._project_onto_manifold(
                gen_climbs, holds_manifold, holds_lookup, return_indices=True
            )
        return gen_climbs
