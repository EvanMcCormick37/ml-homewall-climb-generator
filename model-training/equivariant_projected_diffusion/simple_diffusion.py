import torchaudio
import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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
        self.model = model
        self.timesteps = 100
        self.pred_noise = predict_noise
    
    def loss(self, sample_climbs, cond):
        """Perform a diffusion Training step and return the loss resulting from the model in the training run. Currently returns tuple (loss, real_hold_loss, null_hold_loss)"""
        B = sample_climbs.shape[0]
        S = sample_climbs.shape[1]
        H = sample_climbs.shape[2]
        C = cond.shape[1]
        t = torch.round(torch.rand(B,1), decimals=2)

        noisy = self._forward_diffusion(sample_climbs, t)

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
    
    def _forward_diffusion(self, clean: Tensor, t: Tensor)-> Tensor:
        """Perform forward diffusion to add noise to clean data based on noise adding schedule."""
        a = self._cos_alpha_bar(t)
        return torch.sqrt(a) * clean + torch.sqrt(1-a) * torch.randn_like(clean)
    
    def _cos_alpha_bar(self, t: Tensor)-> Tensor:
        t = t.view(-1,1,1)
        epsilon = 0.0001
        return  torch.cos((t+epsilon)/(1+epsilon)*torch.pi/2)**2
    
    def generate(
        self,
        n: int,
        angle: int,
        grade: int | None = None,
        show_steps: bool = False
    ):
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
        cond = Tensor([[grade/9-0.5 if grade else 0.0, angle/90.0, 1.0, 1.0] for _ in range(n)])

        gen_climbs = torch.randn((n, 20, 4))
        t_tensor = torch.ones((n,1))

        for t in range(1, self.timesteps):
            gen_climbs = self.predict(gen_climbs, cond, t_tensor)
            print('.',end='')
            if t == self.timesteps-1:
                break

            t_tensor -= .01
            gen_climbs = self._forward_diffusion(gen_climbs, t_tensor)
        
        return gen_climbs

class DDPMTrainer():
    def __init__(
        self,
        model: nn.Module,
        dataset: TensorDataset | None = None,
        default_batch_size: int = 64
    ):
        self.model = model
        self.dataset = dataset
        self.default_batch_size = default_batch_size
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    def train(
        self,
        epochs: int,
        save_path: str,
        batch_size: int | None = None,
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

        batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        losses = []

        with tqdm(range(epochs)) as pbar:
            for epoch in pbar:
                total_loss = [0, 0, 0]
                for x, c in batches:
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
