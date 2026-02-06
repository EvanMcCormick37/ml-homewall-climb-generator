import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
from .climb_conversion import ClimbsFeatureScaler
import sqlite3

# Input Data Format:
# climbs: Tensor [Batch_length, 20, 5]
# conditions: Tensor [Batch_length, 4]
# roles: Tensor [Batch_length, 1]
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
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


class Noiser(nn.Module):
    def __init__(self, hidden_dim=128, layers = 5):
        super().__init__()

        self.cond_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.init_conv = ResidualBlock1D(4, hidden_dim, hidden_dim)

        self.residuals = nn.ModuleList([ResidualBlock1D(hidden_dim, hidden_dim, hidden_dim) for _ in range(layers)])

        self.head = nn.Conv1d(hidden_dim, 4, 1)
    
    def forward(self, climbs: Tensor, cond: Tensor, t: Tensor)-> Tensor:
        """
        Run denoising pass. Predicts the added noise from the noisy data.
        
        :param climbs: Tensor with hold-set positions. [B, S, 4]
        :param cond: Tensor with conditional variables. [B, 4]
        :param t: Tensor with timestep of diffusion. [B, 1]
        """
        emb_c = self.cond_mlp(torch.cat([cond,t], dim=1))
        emb_h = self.init_conv(climbs.transpose(1,2), emb_c)

        for layer in self.residuals:
            layer(emb_h, emb_c)

        result = self.head(emb_h).transpose(1,2)

        return result


class ClimbDDPM(nn.Module):
    def __init__(self, model, predict_noise = False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.timesteps = 100
        self.pred_noise = predict_noise
    
    def loss(self, sample_climbs, cond):
        """Perform a diffusion Training step and return the loss resulting from the model in the training run. Currently returns tuple (loss, real_hold_loss, null_hold_loss)"""
        B = sample_climbs.shape[0]
        S = sample_climbs.shape[1]
        H = sample_climbs.shape[2]
        C = cond.shape[1]
        t = torch.round(torch.rand(B,1,device=self.device), decimals=2)

        noisy = self.forward_diffusion(sample_climbs, t)

        pred_clean = self.predict(noisy, cond, t)
        is_real = (sample_climbs[:,:,3] != -2).float().unsqueeze(-1)
        is_null = (sample_climbs[:,:,3] == -2).float().unsqueeze(-1)
        real_hold_loss = self._real_hold_loss(pred_clean, sample_climbs, is_real)
        null_hold_loss = self._null_hold_loss(pred_clean, is_null)
        return real_hold_loss + null_hold_loss, real_hold_loss, null_hold_loss
    
    def predict(self, noisy, cond, t):
        """Return predicted clean data (noisy-pred_noise if the model predicts noise)"""
        prediction = self.model(noisy, cond, t)
        clean = noisy - prediction if self.pred_noise else prediction
        return clean
    
    def _null_hold_loss(self, pred_clean, null_mask):
        """Calculate loss over the null holds"""
        return F.mse_loss(torch.square(pred_clean)*null_mask, null_mask*4)
    
    def _real_hold_loss(self, pred_clean, sample_climbs, real_mask):
        """Get loss over the real holds"""
        return F.mse_loss(pred_clean*real_mask, sample_climbs*real_mask)
    
    def forward_diffusion(self, clean: Tensor, t: Tensor, x_0: Tensor | None = None)-> Tensor:
        """Perform forward diffusion to add noise to clean data based on noise adding schedule."""
        if x_0 is None:
            x_0 = torch.randn_like(clean, device=self.device)
        a = self._cos_alpha_bar(t)
        return torch.sqrt(a) * clean + torch.sqrt(1-a) * x_0
    
    def _cos_alpha_bar(self, t: Tensor)-> Tensor:
        t = t.view(-1,1,1)
        epsilon = 0.0001
        return  torch.cos((t+epsilon)/(1+epsilon)*torch.pi/2)**2
    
    @torch.no_grad()
    def generate(
        self,
        n: int,
        angle: int,
        grade: int | None = None,
        deterministic: bool = False,
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
        cond = torch.tensor([[grade/9-0.5 if grade else 0.0, 0.8, -0.8, angle/90.0] for _ in range(n)], device=self.device)

        x_0 = torch.randn((n, 20, 4), device=self.device)
        t_tensor = torch.ones((n,1), device=self.device)

        for t in range(1, self.timesteps):
            gen_climbs = self.predict(x_0, cond, t_tensor)
            print('.',end='')
            if t == self.timesteps-1:
                return gen_climbs

            t_tensor -= .01
            gen_climbs = self.forward_diffusion(gen_climbs, t_tensor, x_0 if deterministic else None)


class DDPMTrainer():
    def __init__(
        self,
        model: nn.Module,
        dataset: TensorDataset | None = None,
        default_batch_size: int = 64
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataset = dataset
        self.default_batch_size = default_batch_size
        self.optimizer = torch.optim.Adam(model.parameters())
    
    def train(
        self,
        epochs: int,
        save_path: str,
        batch_size: int | None = None,
        num_workers: int | None = None,
        dataset: TensorDataset | None = None,
        save_on_best: bool = False,
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
                total_loss = [0, 0, 0]
                for x, c in batches:
                    x, c = x.to(self.device), c.to(self.device)
                    self.optimizer.zero_grad()
                    loss, real_hold_loss, null_hold_loss = self.model.loss(x, c)
                    loss.backward()
                    self.optimizer.step()

                    # Calc total losses
                    total_loss[0] += loss.item()
                    total_loss[1] += real_hold_loss.item()
                    total_loss[2] += null_hold_loss.item()
                
                improvement = total_loss[0] - losses[-1][0] if len(losses) > 0 else 0
                pbar.set_postfix_str(f"Epoch: {epoch}, Batches:{len(batches)} Total Loss: {total_loss[0]:.2f}, Real Hold Loss: {total_loss[1]:.2f}, Null Hold Loss: {total_loss[2]:.2f}, Improvement: {improvement:.2f}")
                losses.append(total_loss)

                if save_on_best and total_loss > min(losses) and len(losses) % 2 == 0:
                    torch.save(self.model.state_dict(), save_path)
        torch.save(self.model.state_dict(), save_path)
        return self.model, losses


class ClimbDDPMGenerator(nn.Module):
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

        with sqlite3.connect(db_path) as conn:
            holds = pd.read_sql_query("SELECT hold_index, x, y, pull_x, pull_y, useability, is_foot, wall_id FROM holds WHERE wall_id = ?",conn,params=(wall_id,))
            self.holds = self.scaler.transform_hold_features(holds, to_df=True).to_dict('index')

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

        if model_weights_path:
            model.load_state_dict(state_dict=clear_compile_keys(model_weights_path),strict=True)
        if scaler_weights_path:
            self.scaler.load_weights(scaler_weights_path)
    
    def _build_cond_tensor(self, n, grade, diff_scale, angle):
        diff = self.grade_to_diff[diff_scale][grade]
        df_cond = pd.DataFrame({
            "grade": [diff]*n,
            "quality": [2.9]*n,
            "ascents": [100]*n,
            "angle": [angle]*n
        })

        cond = self.scaler.transform_climb_features(df_cond)
        return torch.tensor(cond, device=self.device)
    
    @torch.no_grad()
    def generate(
        self,
        n: int = 1 ,
        angle: int = 45,
        grade: str = 'V4',
        diff_scale: str = 'v_scale',
        deterministic: bool = False,
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

        x_0 = torch.randn((n, 20, 4), device=self.device)
        t_tensor = torch.ones((n,1), device=self.device)

        for t in range(1, self.timesteps):
            gen_climbs = self.model(x_0, cond_t, t_tensor)
            print('.',end='')
            if t == self.timesteps-1:
                return gen_climbs

            t_tensor -= .01
            gen_climbs = self.model.forward_diffusion(gen_climbs, t_tensor, x_0 if deterministic else None)

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