"""
PyTorch model classes and feature engineering for the climb DDPM.

Contains the pure ML components with no database or pool dependencies:
- Noiser (U-Net style denoiser)
- ClimbDDPM (diffusion wrapper)
- ClimbsFeatureScaler (data normalization)
- GRADE_TO_DIFF (grade → difficulty lookup table)
- Tensor utilities (clear_compile_keys, zero_com)
"""
import math
import os
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbeddings(nn.Module):
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


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=padding)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(x)
        h = self.norm1(h)
        gamma, beta = self.cond_proj(cond).unsqueeze(-1).chunk(2, dim=1)
        h = h * (1 + gamma) + beta
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + self.shortcut(x)


class Noiser(nn.Module):
    """U-Net denoiser with learnable null embeddings and zero-COM input projection."""
    def __init__(self, hidden_dim=256, layers=3, in_feature_dim=12, out_feature_dim=12, cond_dim=4, sinusoidal=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.time_mlp = nn.Sequential(
            (SinusoidalPositionEmbeddings(hidden_dim) if sinusoidal else nn.Linear(1, hidden_dim)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.null_cond_emb = nn.Parameter(torch.randn(1, hidden_dim, device=self.device))
        self.combine_t_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.init_conv = ResidualBlock1D(in_feature_dim, hidden_dim, hidden_dim)
        self.down_blocks = nn.ModuleList(
            [ResidualBlock1D(hidden_dim * (i + 1), hidden_dim * (i + 2), hidden_dim) for i in range(layers)]
        )
        self.up_blocks = nn.ModuleList(
            [ResidualBlock1D(hidden_dim * (i + 1), hidden_dim * i, hidden_dim) for i in range(layers, 0, -1)]
        )
        self.head = nn.Conv1d(hidden_dim, out_feature_dim, 1)

    def forward(self, climbs: Tensor, cond: Tensor | None, t: Tensor) -> Tensor:
        (B, S, H) = climbs.shape
        emb_t = self.time_mlp(t)
        emb_c = self.cond_mlp(cond) if cond is not None else self.null_cond_emb.repeat(B, 1)
        emb_c = self.combine_t_mlp(torch.cat([emb_t, emb_c], dim=1))
        emb_h = self.init_conv(climbs.transpose(1, 2), emb_c)

        skip_conns = []
        for layer in self.down_blocks:
            skip_conns.append(emb_h)
            emb_h = layer(emb_h, emb_c)
        for layer in self.up_blocks:
            emb_h = skip_conns.pop() + layer(emb_h, emb_c)

        return self.head(emb_h).transpose(1, 2)


# ---------------------------------------------------------------------------
# DDPM wrapper
# ---------------------------------------------------------------------------

class ClimbDDPM(nn.Module):
    def __init__(self, model: nn.Module, weights_path: Path | str | None = None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        if weights_path:
            self.load_state_dict(clear_compile_keys(weights_path))

    def _cos_alpha_bar(self, t: Tensor) -> Tensor:
        t = t.view(-1, 1, 1)
        epsilon = 0.0004
        return torch.cos((t + epsilon) / (1 + 2 * epsilon) * torch.pi / 2) ** 2

    def _composite_alpha_bar(self, t: Tensor, n_features: int) -> Tensor:
        """Composite schedule that preserves the is_null flag until t=0.8."""
        a_features = self._cos_alpha_bar(t)
        a_null = self._cos_alpha_bar(torch.clamp((t - 0.8) / 0.2, 0.0, 1.0))
        a_full = a_features.expand(-1, -1, n_features - 1)
        return torch.cat([a_full, a_null], dim=2)

    def loss(self, sample_climbs: Tensor, cond: Tensor | None):
        B, S, H = sample_climbs.shape
        t = torch.round(torch.rand(B, 1, device=self.device), decimals=2)
        noise = torch.randn((B, S, H), device=self.device)
        null_mask = (t > 0.8).float()
        noise[:, :, -1] *= null_mask
        noisy = self.forward_diffusion(sample_climbs, t, noise)
        pred_noise = self.model(noisy, cond, t)
        return F.mse_loss(pred_noise, noise)

    def predict_clean(self, noisy, cond, t, epsilon=0.0004):
        (B, S, H) = noisy.shape
        a = self._composite_alpha_bar(t, H)
        prediction = self.model(noisy, cond, t)
        return (noisy - torch.sqrt(1 - a) * prediction) / (torch.sqrt(a) + epsilon)

    def predict_cfg(self, noisy, cond, t, guidance_value=1.0, epsilon=0.0004):
        (B, S, H) = noisy.shape
        a = self._composite_alpha_bar(t, H)
        cf_pred = self.model(noisy, None, t)
        pred = self.model(noisy, cond, t)
        cfg = cf_pred + (pred - cf_pred) * guidance_value
        return (noisy - torch.sqrt(1 - a) * cfg) / (torch.sqrt(a) + epsilon)

    def forward_diffusion(self, clean: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        (B, S, H) = clean.shape
        a = self._composite_alpha_bar(t, H)
        return torch.sqrt(a) * clean + torch.sqrt(1 - a) * noise

    def forward(self, noisy, cond, t):
        return self.predict_clean(noisy, cond, t)


# ---------------------------------------------------------------------------
# Feature scaler
# ---------------------------------------------------------------------------

STYLE_TAGS = ["pinch", "flat"]
SCALED_FEATURES = ["x", "y", "pull_x", "pull_y"]
BINARY_FEATURES = ["is_foot"] + STYLE_TAGS
HOLD_FEATURE_COLS = SCALED_FEATURES + BINARY_FEATURES
NUM_HOLD_FEATURES = len(HOLD_FEATURE_COLS)
COND_FEATURES = ["grade", "quality", "ascents", "angle"]


class ClimbsFeatureScaler:
    def __init__(self, weights_path: str | Path | None = None):
        self.set_weights = False
        self.cond_features_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.hold_scaler = MinMaxScaler(feature_range=(-1, 1))
        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)

    def save_weights(self, path: str | Path):
        joblib.dump({"cond_scaler": self.cond_features_scaler, "hold_scaler": self.hold_scaler}, path)

    def load_weights(self, path: str | Path):
        state = joblib.load(path)
        self.cond_features_scaler = state["cond_scaler"]
        self.hold_scaler = state["hold_scaler"]
        self.set_weights = True

    def transform(self, climbs_to_fit: pd.DataFrame, holds_to_fit: pd.DataFrame):
        scaled_climbs = climbs_to_fit.copy()
        scaled_climbs = self._apply_log_transforms(scaled_climbs)
        if self.set_weights:
            scaled_climbs[COND_FEATURES] = self.cond_features_scaler.transform(scaled_climbs[COND_FEATURES])
        else:
            scaled_climbs[COND_FEATURES] = self.cond_features_scaler.fit_transform(scaled_climbs[COND_FEATURES])

        scaled_holds = holds_to_fit.copy()
        scaled_holds = self._apply_hold_transforms(scaled_holds)
        if self.set_weights:
            scaled_holds[SCALED_FEATURES] = self.hold_scaler.transform(scaled_holds[SCALED_FEATURES])
        else:
            scaled_holds[SCALED_FEATURES] = self.hold_scaler.fit_transform(scaled_holds[SCALED_FEATURES])
            self.set_weights = True

        return scaled_climbs, scaled_holds

    def _apply_log_transforms(self, dfc: pd.DataFrame) -> pd.DataFrame:
        dfc["quality"] -= 3
        dfc["quality"] = np.log(1 - dfc["quality"])
        dfc["ascents"] = np.log(dfc["ascents"])
        return dfc

    def _apply_hold_transforms(self, dfh: pd.DataFrame) -> pd.DataFrame:
        for tag in STYLE_TAGS:
            dfh[tag] = dfh["tags"].apply(
                lambda t: 1.0 if isinstance(t, str) and tag in json.loads(t) else 0.0
            )
        dfh["pull_x"] *= dfh["useability"]
        dfh["pull_y"] *= dfh["useability"]
        for col in BINARY_FEATURES:
            dfh[col] = dfh[col].astype(float)
        return dfh.drop(columns=["useability", "tags"])

    def transform_climb_features(self, climbs_to_transform: pd.DataFrame, to_df: bool = False):
        dfc = climbs_to_transform.copy()
        dfc = self._apply_log_transforms(dfc)
        if to_df:
            dfc[COND_FEATURES] = self.cond_features_scaler.transform(dfc[COND_FEATURES])
        else:
            dfc = self.cond_features_scaler.transform(dfc[COND_FEATURES])
        return dfc

    def transform_hold_features(self, holds_to_transform: pd.DataFrame, to_df: bool = False):
        dfh = holds_to_transform.copy()
        dfh = self._apply_hold_transforms(dfh)
        dfh[SCALED_FEATURES] = self.hold_scaler.transform(dfh[SCALED_FEATURES])
        return dfh if to_df else dfh[HOLD_FEATURE_COLS].values


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clear_compile_keys(filepath: Path | str, map_loc: str = "cpu") -> dict:
    """Strip torch.compile prefixes from state dict keys."""
    state_dict = torch.load(filepath, map_location=map_loc)
    prefix = "_orig_mod."
    return {k.removeprefix(prefix): v for k, v in state_dict.items()}


def zero_com(climbs: Tensor, dim: int) -> Tensor:
    """Zero-Centre-Of-Mass: makes the model translation-invariant."""
    new = climbs.clone()
    com = new[:, :, :dim].mean(dim=1, keepdim=True)
    new[:, :, :dim] -= com
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
        "V1": 13, "V1+": 14, "V2": 15, "V2+": 15.5,
        "V3": 16, "V3+": 17, "V4": 18, "V4+": 19,
        "V5": 20, "V5+": 21, "V6": 22, "V6+": 22.5,
        "V7": 23, "V7+": 23.5, "V8": 24, "V8+": 25,
        "V9": 26, "V9+": 26.5, "V10": 27, "V10+": 27.5,
        "V11": 28, "V11+": 28.5, "V12": 29, "V12+": 29.5,
        "V13": 30, "V13+": 30.5, "V14": 31, "V14+": 31.5,
        "V15": 32, "V15+": 32.5, "V16": 33,
    },
}
