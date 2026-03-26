import torch
from torch import nn, Tensor
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import sqlite3
from torchinfo import summary
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd

from climb_conversion import NUM_ROLES, ClimbsFeatureScaler
from diffusion_utils import (
    GRADE_TO_DIFF,
    DB_PATH,
    SinusoidalPositionEmbeddings,
    clear_compile_keys,
)

#-----------------------------------------------------------------------
# UNET Diffusion Building Blocks
#-----------------------------------------------------------------------

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

        # Residuals for up blocks
        self.up_blocks = nn.ModuleList([ResidualBlock1D(hidden_dim*(i+1), hidden_dim*(i), hidden_dim) for i in range(layers,0,-1)])
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
                
        for layer in self.up_blocks:
            skip_conn = skip_conns.pop()
            emb_h = skip_conn + layer(emb_h, emb_c)
        
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

        # Don't add noise to the is_null flag when t <= 0.8, as the a value for is_null is 0 (this guarantees that the true value of is_null is known by t=0.8)
        null_mask = (t > 0.8).float()

        noise[:,:,-1] *= null_mask
        # print(torch.round(noise,decimals=2))
        noisy = self.forward_diffusion(sample_climbs, t, noise)
        pred_noise = self.model(noisy, cond, t)
        
        return F.mse_loss(pred_noise, noise)
        
    def predict_clean(self, noisy, cond, t, epsilon=.0004):
        """Return predicted clean data."""
        (B, S, H) = noisy.shape
        a = self._composite_alpha_bar(t, H)
        # print(torch.round(a,decimals=2))
        prediction = self.model(noisy, cond, t)
        clean = (noisy - torch.sqrt(1-a)*prediction)/(torch.sqrt(a)+epsilon)
        return clean
    
    def predict_cfg(self, noisy, cond, t, guidance_value=1.0, epsilon=.0004):
        (B, S, H) = noisy.shape
        a = self._composite_alpha_bar(t, H)
        cf_pred = self.model(noisy, None, t)
        pred = self.model(noisy, cond, t)
        cfg = cf_pred+(pred-cf_pred)*guidance_value
        clean = (noisy - torch.sqrt(1-a)*cfg)/(torch.sqrt(a)+epsilon)
        return clean
    
    def forward_diffusion(self, clean: Tensor, t: Tensor, noise: Tensor)-> Tensor:
        """Perform forward diffusion to add noise to clean data based on noise adding schedule."""
        (B, S, H) = clean.shape

        a = self._composite_alpha_bar(t, H)
        # print(torch.round(a,decimals=2))
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
                    loss = self.model.loss(x, c) + self.model.loss(x, None) * 0.25
                    if clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    loss.backward()
                    self.optimizer.step()

                    total_loss+=loss.item()
                total_loss /= len(batches)
                improvement = (losses[-1] - total_loss) if len(losses) > 0 else 0
                pbar.set_postfix_str(f"Epoch: {epoch}, Batch Loss: {total_loss:.2f}, Improvement: {improvement:.2f}, Min Loss: {min(losses) if len(losses) > 0 else 0}, Batches:{len(batches)}")
                losses.append(total_loss)

                if save_on_best and total_loss < min(losses) and len(losses) % 2 == 0:
                    torch.save(self.model.state_dict(), save_path)
            self.scheduler.step()
        if save_path is not None:
            torch.save(self.model.state_dict(), save_path)
        return self.model, losses

# ---------------------------------------------------------------------------
# ClimbDDPMGenerator
# ---------------------------------------------------------------------------
FEATURE_WEIGHTS = [1.0,1.0,1.0,1.0,0.5,0.5,0.5]

class ProjectedDiffusionGenerator():
    """Class for handling climb projection logic. Simplifies ClimbDDPMGenerator logic by extracting Projected diffusion state+functions."""
    def __init__(self, scaler: ClimbsFeatureScaler, feature_weights = torch.tensor(FEATURE_WEIGHTS)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = scaler
        self.holds_manifolds = {}
        self.holds_lookup = {}
        self.feature_weights = feature_weights
        with sqlite3.connect(DB_PATH) as conn:
            holds = pd.read_sql_query("SELECT hold_index, x, y, pull_x, pull_y, useability, is_foot, layout_id FROM holds", conn)
            layout_ids = list(set(holds['layout_id'].values))
        
        scaled_holds = self.scaler.transform_hold_features(holds, to_df=True)
        
        for layout_id in layout_ids:
            df = scaled_holds[scaled_holds['layout_id']==layout_id]
            self.holds_manifolds[layout_id] = torch.tensor(df[['x','y','pull_x','pull_y']].values, dtype=torch.float32)
            self.holds_lookup[layout_id] = df['hold_index'].values
            self.holds_lookup[layout_id] = self.holds_lookup[layout_id]
    
    def log_hold_means(self, layout_id: str | None = None):
        """Log the hold means for each wall."""
        for k, manifold in self.holds_manifolds.items():
            if layout_id == None or layout_id == k:
                means = torch.mean(manifold, dim=0)
                print(f"Wall-id--{k}; Means-- x:{means[0].item()}, y:{means[1].item()}, Px:{means[2].item()}, Py:{means[3].item()} ")
    
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
        dists = torch.cdist(flat_climbs[:,:,:(H-NUM_ROLES)]*self.feature_weights, offset_manifold*self.feature_weights)
        idx = dists.argmin(dim=1)
        return offset_manifold[idx].reshape(B, S, -1)
    
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
        offset_manifold: Tensor,      # (M, H - NUM_ROLES)
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
        nonnull_mask = (gen_climbs[:,:,-1] < 0.95).float()
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
        dists = torch.cdist(flat[:,:,:(H-NUM_ROLES)]*self.feature_weights, offset_manifold*self.feature_weights)              # (G*B, S, M)
        nn_dists = dists.min(dim=2).values                      # (G*B, S)
        batch_dist = nn_dists.reshape(G, B, S)                  # (G, B, S)
        batch_dist = nonnull_mask.unsqueeze(0) * batch_dist

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
        translation[:, 0, 1] = best_dy   # y col  (pull_x/pull_y cols 2,3 left alone)
        translated_climbs = gen_climbs + translation

        # Now do the standard index projection on the translated climbs
        return self._project_onto_indices(translated_climbs, offset_manifold, layout_id)

    def _project_onto_indices(
            self,
            gen_climbs: Tensor,
            offset_manifold: Tensor,
            layout_id: str,
        ):
        """Project climb onto the final hold indices (and remove null holds)"""
        
        B, S, H = gen_climbs.shape

        flat_climbs = gen_climbs.reshape(-1,H)                  # (B*S, H)

        dists = torch.cdist(flat_climbs[:,:,:(H-NUM_ROLES)]*self.feature_weights, offset_manifold*self.feature_weights)       # (B*S, H, M)
        idx = dists.argmin(dim=1)
        holds = self.holds_lookup[layout_id][idx]
        holds = holds.reshape(B, S)
        
        roles = np.argmax(gen_climbs[:,:,(H-NUM_ROLES):].detach().numpy(),axis=2)
        climbs = np.stack([holds, roles], axis=2)
        print(f"climbs.shape: {climbs.shape}")
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

class ClimbDDPMGenerator():
    """"""
    def __init__(
            self,
            scaler: ClimbsFeatureScaler,
            ddpm: ClimbDDPM,
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = scaler
        self.projector = ProjectedDiffusionGenerator(scaler)
        self.ddpm = ddpm
        self._cond_cache = {}
        self.deterministic_noise_generator = torch.Generator(device=self.device)
    
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
        seed: int,
    )->list[list[list[int]]]:
        """
        Generate a climb or batch of climbs with the given conditions using the standard DDPM iterative denoising process.

        :param layout_id: The Wall-ID on which to generate the climb.
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
        :return: A set of generated climbs according to the specified 
        :rtype: list[list[list[int]]]
        """
        # Seed Noise Generator
        if deterministic:
            self.deterministic_noise_generator.manual_seed(seed)
        
        # Handle manifold offset
        auto = True if x_offset is None else False
        offset_manifold = self.projector._get_offset_manifold(layout_id, x_offset)

        # CORE LOGIC
        cond_t = self._build_cond_tensor(n, grade, diff_scale, angle)
        x_t = torch.randn((n, 20, 11), device=self.device, generator=self.deterministic_noise_generator) if deterministic else torch.randn((n, 20, 11), device=self.device)
        noisy = x_t.clone()
        t_tensor = torch.ones((n,1), device=self.device)
        
        for _ in range(0, timesteps):
            gen_climbs = self.ddpm(noisy, cond_t, t_tensor)

            if t_tensor[0].item() < t_start_projection: # This block might be problematic. If not projecting, don't enter it at all.
                alpha_p = self.projector._projection_strength(t_tensor, t_start_projection)
                projected_climbs = self.projector._project_onto_manifold(gen_climbs, offset_manifold)
                gen_climbs = alpha_p*(projected_climbs) + (1-alpha_p)*(gen_climbs)
            
            t_tensor -= 1.0/timesteps
            noisy = self.ddpm.forward_diffusion(gen_climbs, t_tensor, x_t if deterministic else torch.randn_like(x_t))
        
        if auto:
            return self.projector._project_onto_indices_with_translation(gen_climbs, offset_manifold, layout_id)
        return self.projector._project_onto_indices(gen_climbs, offset_manifold, layout_id)