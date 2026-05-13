"""
ClimbDDPMGenerator and GeneratorPool.

Orchestrates inference: acquires a generator from the pool, runs the DDPM
denoising loop with manifold projection, and returns hold-index results.

Model classes and feature engineering live in ddpm_model.py.
"""
import sqlite3
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from app.config import settings
from app.services.utils.ddpm_model import (
    ClimbDDPM,
    ClimbsFeatureScaler,
    Noiser,
)


# ---------------------------------------------------------------------------
# ClimbDDPMGenerator
# ---------------------------------------------------------------------------

class ClimbDDPMGenerator:
    HAND_FEATURE_WEIGHTS = torch.tensor([1.0, 1.0, 0.8, 0.8, 2.0, 0.1, 0.1])
    FOOT_FEATURE_WEIGHTS = torch.tensor([1.0, 1.0, 0.05, 0.05, 1.0, 0.05, 0.05])
    NUM_ROLES = 5
    NUM_FEATURES = 7
    START_ROLE = 0
    FINISH_ROLE = 1
    HAND_ROLE = 2
    FOOT_ROLE = 3

    def __init__(self, scaler: ClimbsFeatureScaler, ddpm: ClimbDDPM):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = scaler
        self.ddpm = ddpm
        self._cond_cache: dict = {}
        self.holds_manifolds: dict = {}
        self.holds_lookup: dict = {}
        self.holds_com: dict = {}
        self.deterministic_noise_generator = torch.Generator(device=self.device)
        self.update_hold_manifolds()

    def update_hold_manifolds(self):
        with sqlite3.connect(settings.DB_PATH) as conn:
            holds = pd.read_sql_query(
                "SELECT hold_index, x, y, pull_x, pull_y, useability, is_foot, tags, layout_id FROM holds",
                conn,
            )
            layout_ids = holds["layout_id"].unique()

        scaled_holds = self.scaler.transform_hold_features(holds, to_df=True)
        self.holds_manifolds.clear()
        self.holds_lookup.clear()
        self.holds_com.clear()  # ← new

        for layout_id in layout_ids:
            df = scaled_holds[scaled_holds["layout_id"] == layout_id]
            manifold = torch.tensor(
                df[["x", "y", "pull_x", "pull_y", "is_foot", "pinch", "flat"]].values,
                dtype=torch.float32,
            )
            self.holds_manifolds[layout_id] = manifold
            self.holds_com[layout_id] = manifold.mean(dim=0)  # ← new
            self.holds_lookup[layout_id] = df["hold_index"].values
    def log_hold_means(self, layout_id: str | None = None):
        for k, manifold in self.holds_manifolds.items():
            if layout_id is None or layout_id == k:
                means = torch.mean(manifold, dim=0)
                print(f"layout-id--{k}; Means-- x:{means[0].item()}, y:{means[1].item()}")

    def _build_cond_tensor(self, n, diff, angle):
        cache_key = (diff, angle)
        if cache_key not in self._cond_cache:
            row = np.array([[diff, 3.0, 1000, float(angle)]])
            scaled = self.scaler.transform_climb_features(
                pd.DataFrame(row, columns=["grade", "quality", "ascents", "angle"])
            )
            self._cond_cache[cache_key] = scaled
        return torch.tensor(np.tile(self._cond_cache[cache_key], (n, 1)), device=self.device, dtype=torch.float32)

    def _get_climb_translation(self, layout_id: str, x_offset: float | None) -> Tensor:
        """
        Compute the (x, y) translation to apply to generated climbs so they
        align with the unshifted hold manifold before projection.
        Returns a (1, 1, H) tensor with the translation in dims 0 and 1 only.
        """
        manifold = self.holds_manifolds[layout_id]
        com = self.holds_com[layout_id]
        if x_offset is None:
            x_offset = torch.clamp(torch.randn(size=(1,)), -1.0, 1.0).item() / 2
        x_range = (manifold[:, 0].max() - manifold[:, 0].min()).item()
        x_translate = x_offset * x_range / 2 + com[0].item()
        y_translate = com[1].item()
        H = self.NUM_FEATURES + self.NUM_ROLES
        t = torch.zeros(1, 1, H, device=self.device)
        t[0, 0, 0] = x_translate
        t[0, 0, 1] = y_translate
        
        return t
        
    def _projection_strength(self, t: Tensor, t_start_projection: float = 0.8):
        assert t_start_projection <= 0.8
        a = (t_start_projection - t) / t_start_projection
        if t_start_projection > 0.5:
            strength = 1 + torch.cos((a + 0.5) * torch.pi)
        else:
            strength = 1 - torch.cos(a * torch.pi / 2)
        return torch.where(t > t_start_projection, torch.zeros_like(t), strength).unsqueeze(2)

    def _get_nearest_manifold_neighbors(self, flat_climbs: Tensor, offset_manifold: Tensor) -> tuple[Tensor, Tensor]:
        is_hand_mask = flat_climbs[:, -2] < 0.85
        is_foot_mask = ~is_hand_mask
        features = flat_climbs[:, : self.NUM_FEATURES]
        N = flat_climbs.shape[0]
        idx = torch.empty(N, dtype=torch.long, device=flat_climbs.device)
        min_dists = torch.empty(N, dtype=torch.float32, device=flat_climbs.device)
        if is_hand_mask.any():
            hand_dists = torch.cdist(
                features[is_hand_mask] * self.HAND_FEATURE_WEIGHTS.unsqueeze(0),
                offset_manifold * self.HAND_FEATURE_WEIGHTS.unsqueeze(0),
            )
            vals, indices = hand_dists.min(dim=1)
            idx[is_hand_mask] = indices
            min_dists[is_hand_mask] = vals
        if is_foot_mask.any():
            foot_dists = torch.cdist(
                features[is_foot_mask] * self.FOOT_FEATURE_WEIGHTS.unsqueeze(0),
                offset_manifold * self.FOOT_FEATURE_WEIGHTS.unsqueeze(0),
            )
            vals, indices = foot_dists.min(dim=1)
            idx[is_foot_mask] = indices
            min_dists[is_foot_mask] = vals
        return min_dists, idx

    def _project_onto_manifold(self, gen_climbs: Tensor, offset_manifold: Tensor) -> Tensor:
        B, S, H = gen_climbs.shape
        flat_climbs = gen_climbs.reshape(-1, H)
        null_mask = (flat_climbs[:, -1] < 0.95)
        _, idx = self._get_nearest_manifold_neighbors(flat_climbs, offset_manifold)
        projected_features = offset_manifold[idx] * null_mask.unsqueeze(1)
        return torch.cat([projected_features.reshape(B, S, -1), gen_climbs[:, :, self.NUM_FEATURES:]], dim=2)

    def _project_onto_indices(self, gen_climbs: Tensor, offset_manifold: Tensor, layout_id: str) -> list[list[list[int]]]:
        B, S, H = gen_climbs.shape
        roles = torch.argmax(gen_climbs[:, :, self.NUM_FEATURES:], dim=2).detach().cpu().numpy()
        flat_climbs = gen_climbs.reshape(-1, H)
        _, idx = self._get_nearest_manifold_neighbors(flat_climbs, offset_manifold)
        y_vals = offset_manifold[idx, 1].detach().cpu().numpy().reshape(B, S)
        if not isinstance(self.holds_lookup[layout_id], torch.Tensor):
            idx = idx.cpu().numpy()
        holds = self.holds_lookup[layout_id][idx].reshape(B, S)
        if isinstance(holds, torch.Tensor):
            holds = holds.detach().cpu().numpy()
        climbs = np.stack([holds, roles, y_vals], axis=2)
        deduped_climbs = []
        for c in climbs:
            valid_mask = c[:, 1] != 4
            c_valid = c[valid_mask]
            c_sorted = c_valid[c_valid[:, 1].argsort()]
            _, unique_indices = np.unique(c_sorted[:, 0], return_index=True)
            c_deduped = c_sorted[unique_indices]
            if not np.any(c_deduped[:, 1] == self.FINISH_ROLE):
                c_deduped[np.argmax(c_deduped[:, 2]), 1] = self.FINISH_ROLE
            if not np.any(c_deduped[:, 1] == self.START_ROLE):
                min_y_idx = np.argmin(c_deduped[:, 2] + (c_deduped[:, 1] == self.FOOT_ROLE).astype(np.float32))
                if min_y_idx != np.argmax(c_deduped[:, 2]) or len(c_deduped) == 1:
                    c_deduped[min_y_idx, 1] = self.START_ROLE
            deduped_climbs.append(c_deduped[:, :2].astype(int).tolist())
        return deduped_climbs
    
    def _measure_fitness(self, gen_climbs: Tensor, manifold: Tensor) -> Tensor:
        """
        Fitness score for each climb: mean nearest-manifold Euclidean distance
        across non-null holds. Lower score = better fit.

        Args:
            gen_climbs: (B, S, H) in manifold space (translation already applied).
            manifold:   unshifted hold manifold for this layout.

        Returns:
            (B,) float tensor of per-climb fitness scores.
        """
        B, S, H = gen_climbs.shape
        flat = gen_climbs.reshape(-1, H)

        # Non-null holds: is_null feature (last dim) below threshold
        real_hold_mask = (flat[:, -1] < 0.95).float()  # (B*S,)

        min_dists, _ = self._get_nearest_manifold_neighbors(flat, manifold)

        # Mean distance over real holds only; clamp denominator to avoid div-by-zero
        weighted_dists = (min_dists * real_hold_mask).reshape(B, S).sum(dim=1)
        real_hold_counts = real_hold_mask.reshape(B, S).sum(dim=1).clamp(min=1)
        return weighted_dists / real_hold_counts  # (B,)
    
    @torch.no_grad()
    def generate(self, layout_id, n, angle, difficulty, timesteps, deterministic,
                t_start_projection, x_offset, guidance_value, seed):
        if deterministic:
            self.deterministic_noise_generator.manual_seed(seed)

        manifold = self.holds_manifolds[layout_id]          # unshifted
        translation = self._get_climb_translation(layout_id, x_offset)  # (1,1,H)

        diff = difficulty
        if diff > 22:
            guidance_value *= 0.5
        cond_t = self._build_cond_tensor(n, diff, angle)

        shape = (n, 20, self.NUM_ROLES + self.NUM_FEATURES)
        x_t = (
            torch.randn(shape, device=self.device, generator=self.deterministic_noise_generator)
            if deterministic else torch.randn(shape, device=self.device)
        )
        noisy = x_t.clone()
        t_tensor = torch.ones((n, 1), device=self.device)

        for _ in range(timesteps):
            gen_climbs = self.ddpm.predict_cfg(noisy, cond_t, t_tensor, guidance_value)

            if t_tensor[0].item() < t_start_projection:
                alpha_p = self._projection_strength(t_tensor, t_start_projection)
                # Translate into manifold space, project, translate back
                projected = self._project_onto_manifold(gen_climbs + translation, manifold)
                gen_climbs = alpha_p * (projected - translation) + (1 - alpha_p) * gen_climbs

            t_tensor -= 1.0 / timesteps
            noisy = self.ddpm.forward_diffusion(
                gen_climbs, t_tensor,
                noisy if deterministic else torch.randn_like(noisy),
            )

        # Translate into manifold space for the final discrete projection
        gen_climbs_manifold = gen_climbs + translation
        return self._project_onto_indices(gen_climbs_manifold, manifold, layout_id)

    @torch.no_grad()
    def generate_evolutionary(
        self,
        layout_id: str,
        n: int,
        p: int,
        angle: int,
        difficulty: float,
        timesteps: int,
        deterministic: bool,
        x_offset: float | None,
        guidance_value: float,
        seed: int,
    ) -> list[list[list[int]]]:
        """
        Evolutionary reverse-diffusion generation.

        Maintains n independent pools of p candidates each — shape (n, p, S, H).
        At every step (including the last), fitness is evaluated over each pool of p
        and the single best candidate per pool is selected as the survivor.
        On non-final steps the survivor is tiled back to p and re-noised.
        On the final step the n survivors are projected onto hold indices and returned.

        Args:
            n: Number of climbs to return (one independent pool each).
            p: Candidates per pool evaluated at each diffusion step.
        """
        if deterministic:
            self.deterministic_noise_generator.manual_seed(seed)

        manifold = self.holds_manifolds[layout_id]
        translation = self._get_climb_translation(layout_id, x_offset)  # (1, 1, H)

        if difficulty > 22:
            guidance_value *= 0.5

        # cond_t: (n, cond_dim) -> (n*p, cond_dim) with each row repeated p times
        cond_t = self._build_cond_tensor(n, difficulty, angle)           # (n, cond_dim)
        cond_t_flat = cond_t.repeat_interleave(p, dim=0)                 # (n*p, cond_dim)

        S = 20
        H = self.NUM_FEATURES + self.NUM_ROLES
        noisy = (
            torch.randn((n, p, S, H), device=self.device,
                        generator=self.deterministic_noise_generator)
            if deterministic
            else torch.randn((n, p, S, H), device=self.device)
        )

        t_val = 1.0
        survivors: Tensor | None = None

        for step in range(timesteps):
            t_flat = torch.full((n * p, 1), t_val, device=self.device)

            # Denoise — DDPM requires (B, S, H)
            gen_climbs_flat = self.ddpm.predict_cfg(
                noisy.reshape(n * p, S, H), cond_t_flat, t_flat, guidance_value
            )                                                            # (n*p, S, H)
            gen_climbs = gen_climbs_flat.reshape(n, p, S, H)

            t_val -= 1.0 / timesteps

            # Fitness over each pool — lower is better
            # translation (1,1,H) broadcasts to (n,p,S,H) then flattened to (n*p,S,H)
            fitness = self._measure_fitness(
                (gen_climbs + translation).reshape(n * p, S, H), manifold
            ).reshape(n, p)                                              # (n, p)

            best_idx = fitness.argmin(dim=1)                            # (n,)
            survivors = gen_climbs[
                torch.arange(n, device=self.device), best_idx
            ]                                                            # (n, S, H)

            if step == timesteps - 1:
                break

            # Tile survivors → (n, p, S, H) and re-noise for the next step
            t_next_flat = torch.full((n * p, 1), t_val, device=self.device)
            noisy = self.ddpm.forward_diffusion(
                survivors.unsqueeze(1).repeat(1, p, 1, 1).reshape(n * p, S, H),
                t_next_flat,
                torch.randn(n * p, S, H, device=self.device),
            ).reshape(n, p, S, H)

        # survivors: (n, S, H) — best from each pool after final fitness selection
        # translation (1,1,H) broadcasts to (n,S,H)
        gen_climbs_manifold = survivors + translation
        return self._project_onto_indices(gen_climbs_manifold, manifold, layout_id)


# ---------------------------------------------------------------------------
# Generator pool
# ---------------------------------------------------------------------------

def reset_generator() -> ClimbDDPMGenerator:
    scaler = ClimbsFeatureScaler(weights_path=settings.SCALER_WEIGHTS_PATH)
    ddpm = ClimbDDPM(model=Noiser(), weights_path=settings.DDPM_WEIGHTS_PATH)
    ddpm.eval()
    gen = ClimbDDPMGenerator(scaler=scaler, ddpm=ddpm)
    gen.log_hold_means()
    return gen


class GeneratorPool:
    """Pool of ClimbDDPMGenerator instances for concurrent generation.

    Each slot owns its own model copy so concurrent requests never share
    mutable PyTorch state. Callers block on acquire() when all slots are busy.
    """

    def __init__(self, size: int):
        from queue import Queue
        self._pool: "Queue[ClimbDDPMGenerator]" = Queue(maxsize=size)
        for _ in range(size):
            self._pool.put(reset_generator())

    @contextmanager
    def acquire(self):
        gen = self._pool.get(block=True)
        try:
            yield gen
        finally:
            self._pool.put(gen)

    def update_all_hold_manifolds(self):
        """Refresh hold manifolds on every pooled generator instance."""
        items = []
        while not self._pool.empty():
            items.append(self._pool.get_nowait())
        for gen in items:
            gen.update_hold_manifolds()
            self._pool.put(gen)


generator_pool = GeneratorPool(size=settings.GENERATOR_POOL_SIZE)
