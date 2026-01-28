import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from dataclasses import dataclass, field
from torch_geometric.nn import knn_graph
from typing import Protocol
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

@dataclass
class LossWeights:
    pos: float = 1.0
    vec: float = 1.0
    scalars: float = 1.0
    roles: float = 1.0

@dataclass
class ClimbDiffusionConfig:
    num_scalar_features: int = 6
    num_roles: int = 5
    null_token: int = 4
    embedding_dim: int = 128
    num_layers: int = 6
    k: int = 12
    schedule_offset: float = 0.008
    schedule_scale: float = 1.008
    loss_weights: LossWeights = field(default_factory=LossWeights)

@dataclass
class ClimbPredictions:
    x: Tensor
    v: Tensor
    s: Tensor
    r: Tensor

class ClimbBatch(Protocol):
    """Extends torch_geometry Batch class for representing climbs as point clouds. Each climb is padded to 30 nodes by adding random holds with NULL role."""
    pos: Tensor
    vec: Tensor
    scalars: Tensor
    roles: Tensor
    batch: Tensor
    num_graphs: int
    nodes_per_graph: int = 30
    
class GravEGNNConv(nn.Module):
    """Equivariant Graph Neural Network layer conditioned for Gravity Sensitivity. O(2)+R^3 Equivariant (Translations in 3D space + Rotations and Reflections around the Z-axis."""
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int
    ):
        super().__init__()

        # 1. Message MLP: Process node features and geometric info:
        # Message = Phi(h_i, h_j, squared_dist, z_diff, edge_props)
        msg_input_dim = (node_dim*2)+1+1+edge_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # 2. Node MLP: Updates scalar node features
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim+hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        # 3. Coordinate Weights: Predicts magnitude of coordinate update based on messages
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 4. Vector Weights: Predicts magnitude of vector (Pull-direction) update
        self.vector_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Gates vector update based on current state
        self.vector_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        v: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Perform one forward pass through the layer."""
        row, col = edge_index
        
        # Calculate Squared Dist
        rel_pos = x[row]-x[col]
        squared_dist = torch.sum(rel_pos**2, dim=-1, keepdim=True)

        # Calculate Gravity-Sensitive Metric: Z-Diff
        z_diff = (x[row][:,2]-x[col][:,2]).unsqueeze(-1)

        if edge_attr is None:
            edge_attr = torch.zeros(rel_pos.size(0), 0, device = x.device)
   
        # Compute Message
        row, col = edge_index
        msg = self.msg_mlp(torch.cat([h[row], h[col], squared_dist, z_diff, edge_attr],dim=-1))
        
        # Compute Updated Weights from message. Normalize by dividing by the squared distance to prevent "Point ejection"
        x_weights = self.coord_mlp(msg) / (squared_dist + 1e-8)
        v_weights = self.vector_mlp(msg) / (squared_dist + 1e-8)

        # Aggregate Messages for Scalar feature updates
        aggr_msg = torch_scatter.scatter(msg, row, dim=0, reduce='sum', dim_size=x.size(0))

        # Update points by relative position (points 'push'/'pull' their neighbors). This update is O(3) equivariant.
        weighted_rel_pos = rel_pos * x_weights
        x_update = torch_scatter.scatter(weighted_rel_pos, row, dim=0, reduce='sum', dim_size = x.size(0))
        x_new = x + x_update

        # Update hold directions in the same manner, using the vector weights instead of position weights.
        weighted_rel_pos = rel_pos * v_weights
        v_update = torch_scatter.scatter(weighted_rel_pos, row, dim=0, reduce='sum', dim_size = v.size(0))
        v_new = v + v_update

        # Update Scalar Features
        h_new = h + self.node_mlp(torch.cat([h, aggr_msg], dim=-1))


        return h_new, x_new, v_new


class ClimbDiffusionModel(nn.Module):
    """Denoising Diffusion Probabalistic Model (DDPM) for generating climbs as point clouds. EGNN trunk for permutation+O(2)+R^3 equivariance, Discrete Absorbing State for hold categories (including masked NULL state) allowing for varying cardinality."""
    def __init__(self, config: ClimbDiffusionConfig = ClimbDiffusionConfig()):
        super().__init__()

        #Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim)
        )

        #Role Embedding
        self.role_embedding = nn.Embedding(config.num_roles, config.embedding_dim)

        #Scalar Feature Embedding
        self.scalar_feature_embedding = nn.Linear(config.num_scalar_features, config.embedding_dim)

        # Root Layer (Combine Time, Role, Scalar Embeddings)
        self.roots = nn.Sequential(
            nn.Linear(config.embedding_dim*3,config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim,config.embedding_dim)
        )

        # -- Trunk -- EGNN convolutional layer * n_layers
        self.trunk = nn.ModuleList([
            GravEGNNConv(
                node_dim = config.embedding_dim,
                edge_dim = 0,
                hidden_dim = config.embedding_dim
            ) for _ in range(config.num_layers)
        ])

        # Output Heads (Scalar Feature Prediction, Role Prediction)
        self.feature_head = nn.Linear(config.embedding_dim, config.num_scalar_features)
        self.role_head = nn.Linear(config.embedding_dim, config.num_roles)

    def forward(
        self,
        x: Tensor,
        v: Tensor,
        scalars: Tensor,
        roles: Tensor,
        t: Tensor,
        batch: Tensor
    ):
        """Perform one forward pass through the model. Returns a dictionary of climbing predictions."""
        # Embed time, features and combine them in roots.
        t_emb = self.time_mlp(t.unsqueeze(-1))
        t_nodes = t_emb[batch]
        
        role_emb = self.role_embedding(roles)
        scalar_emb = self.scalar_feature_embedding(scalars)

        
        # initial node-state h
        h = self.roots(torch.cat([scalar_emb, role_emb, t_nodes],dim=-1))
        
        # Construct the Graph (K-NN)
        edge_index = knn_graph(x,k=12,batch=batch)

        
        # Pass the data through the trunk
        for layer in self.trunk:
            h, x, v = layer(h, x, v, edge_index)
        
        # Decode scalar and role heads
        r = self.role_head(h)
        s = self.feature_head(h)
        return ClimbPredictions(
            x=x,
            v=v,
            s=s,
            r=r
        )
        
class ClimbDiffusionTrainer(nn.Module):
    """Trainer for the Climbing Diffusion Model"""
    def __init__(
        self,
        model: ClimbDiffusionModel,
        config: ClimbDiffusionConfig = ClimbDiffusionConfig(),
    ):
        super().__init__()
        self.model = model
        self.config = config

    def _cosine_alpha_bar(self, t: Tensor) -> Tensor:
        """Compute alpha bar using cosine schedule."""
        o = self.config.schedule_offset
        s = self.config.schedule_scale

        return torch.cos((t+o)/s*torch.pi/2)**2
    
    def _forward_diffusion_continuous(
        self,
        clean: Tensor,
        noise: Tensor,
        alpha_bar: Tensor
    ) -> Tensor:
        """
        Apply the forward diffusion (noising) process by adding gaussian noise to the clean data features.
        
        :param clean: The clean data
        :type clean: Tensor
        :param noise: The Gaussian noise
        :type noise: Tensor
        :param alpha_bar: The noising schedule values
        :type alpha_bar: Tensor
        :return: The data with added noise.
        :rtype: Tensor
        """
        alpha_bar = alpha_bar.unsqueeze(-1)
        return torch.sqrt(alpha_bar) * clean + torch.sqrt(1-alpha_bar)*noise

    def _forward_diffusion_discrete(
        self,
        roles: Tensor,
        probability_mask: Tensor
    ):
        """
        Apply discrete absorbing diffusion to the roles.
        
        :param roles: The one-hot encoded roles (includes the NULL state)
        :type roles: Tensor
        :param probability_mask: Transition probabilities for each node.
        :type probability_mask: Tensor
        :return: The new discrete roles state. Roles randomly transition to the NULL state during the forward process. NULL roles do not transition out of the null state.
        :rtype: Tensor
        """

        roles_noised = roles.clone()

        mask_decision = torch.rand_like(probability_mask) < probability_mask
        roles_noised[mask_decision] = self.config.null_token

        roles_noised[roles==self.config.null_token] = self.config.null_token

        return roles_noised
    
    def get_loss(
        self,
        batch: ClimbBatch
    ) -> dict[str, Tensor]:
        """
        Perform the forward diffusion process, predict the clean data, and compute the loss function across all feature categories.
        
        :param batch: torch_geometric Batch, stores individual training examples as node graphs within one large, disconnected node graph.

                **Batch Features** (N is number of nodes)
                - pos:              Positions [N,3]
                - vec:              Pull Vectors [N,3]
                - scalars:          Scalar Features [N, self.config.num_scalar_features]
                - roles:            Categorical Role [N]
                - num_graphs        (Int) Number of node graphs in the batch
                - batch:            Batch Index (Tracks which nodes belong to which node graph)
        :type batch: ClimbBatch
        :return: Dict of model loss by type (pos, vec, scalars, roles, weighted_sum)
        :rtype: dict[str, Tensor]
        """

        # Set time and alpha_bar
        t = torch.rand(batch.num_graphs, device=batch.pos.device)
        t_nodes = t[batch.batch]
        a = self._cosine_alpha_bar(t_nodes)

        # Create Gaussian noise over continuous features
        noise_x = torch.randn_like(batch.pos)
        noise_v = torch.randn_like(batch.vec)
        noise_s = torch.randn_like(batch.scalars)

        # Apply forward diffusion
        x_noised = self._forward_diffusion_continuous(batch.pos, noise_x, a)
        v_noised = self._forward_diffusion_continuous(batch.vec, noise_v, a)
        s_noised = self._forward_diffusion_continuous(batch.scalars, noise_s, a)
        r_noised = self._forward_diffusion_discrete(batch.roles, t_nodes)

        # Predict the clean data (denoising step)
        preds = self.model(x_noised, v_noised, s_noised, r_noised, t, batch.batch)

        # Create a boolean mask to prevent null hold positions from affecting the model's accuracy
        is_real = (batch.roles != self.config.null_token).float().unsqueeze(-1)

        # Calculate model loss
        x_loss = F.mse_loss(preds.x * is_real, batch.pos * is_real)
        v_loss = F.mse_loss(preds.v * is_real, batch.vec * is_real)
        s_loss = F.mse_loss(preds.s * is_real, batch.scalars * is_real)
        r_loss = F.cross_entropy(preds.r, batch.roles)

        w = self.config.loss_weights
        weighted_sum = sum([
            w.pos * x_loss,
            w.vec * v_loss,
            w.scalars * s_loss,
            w.roles * r_loss,
        ], torch.tensor([0]))

        return {
            "pos": x_loss,
            "vec": v_loss,
            "scalars": s_loss,
            "roles": r_loss,
            "weighted_sum": weighted_sum,
        }
    
class ClimbDiffusionSampler:
    
    """
    DDPM generation model for generating climbs as point clouds.
    
    :param model: Trained diffusion model
    :type model: ClimbDiffusionModel
    :param config: Configuration settings
    :type config: ClimbClimbDiffusionConfig
    :param num_steps: Number of diffusion->denoising steps to run during generation
    :type num_steps: int
    """

    def __init__(
        self,
        model: ClimbDiffusionModel,
        config: ClimbDiffusionConfig = ClimbDiffusionConfig(),
        nodes_per_sample: int = 30,
        device: str = 'cpu',
        num_steps: int = 100
    ):
        self.model = model
        self.device = device
        self.config = config
        self.nodes_per_sample = nodes_per_sample
        self.num_steps = num_steps

    def _cosine_alpha_bar(self, t):
        """Compute alpha bar using cosine schedule."""
        o = self.config.schedule_offset
        s = self.config.schedule_scale

        return torch.cos((t+o)/s*torch.pi/2)**2
    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 1
    ) -> ClimbPredictions:
        """Generate *num_samples* sample climbs using the ClimbDiffusionModel. Vanilla generation which generates climbs in 3d space but doesn't perform projected/guided diffusion."""
        self.model.eval()

        num_nodes = self.nodes_per_sample * num_samples

        # Initialize climbs as random noise
        x = torch.randn(num_nodes, 3)
        v = torch.randn(num_nodes, 3)
        s = torch.randn(num_nodes, self.config.num_scalar_features)
        r = torch.full((num_nodes,),self.config.null_token)

        # Assign nodes to batches
        batch = torch.arange(num_samples).repeat_interleave(self.nodes_per_sample)
        
        # Create reverse diffusion timesteps
        timesteps = torch.linspace(1,0, self.num_steps+1)

        denoised = None
        # Perform Reverse Diffusion
        for i in range(self.num_steps):
            # Calculate t and alpha bar
            t = timesteps[i]
            t_next = timesteps[i+1]
            t_batch = t.expand(num_samples)

            # Predict denoised climbs
            denoised = self.model(x, v, s, r, t_batch, batch)

            # Calculate alpha bar from t value
            a = self._cosine_alpha_bar(t_next)
            
            # Add noise back in according to alpha bar schedule
            x = torch.sqrt(a) * denoised.x + torch.sqrt(1-a) * torch.randn_like(denoised.x)
            v = torch.sqrt(a) * denoised.v + torch.sqrt(1-a) * torch.randn_like(denoised.v)
            s = torch.sqrt(a) * denoised.s + torch.sqrt(1-a) * torch.randn_like(denoised.s)
            
            # Randomly remask some nodes to return them to NULL according to schedule
            mask_decision = torch.rand(r.size(0)) < t_next
            r_logits = denoised.r
            r_logits[:,self.config.null_token] -= 6
            r = torch.argmax(r_logits, dim=-1)
            r[mask_decision] = self.config.null_token

        assert denoised is not None
        return denoised
    
# -----------------------------------------------------------------------------------
# Square Recognition Model (Toy Model): E(2) EGNN
# -----------------------------------------------------------------------------------

class E2_EGNN(nn.Module):
    """Equivariant Graph Neural Network layer"""

    def __init__(self, emb_dim, hidden_dim, activation_func = nn.ReLU()):
        # h_i + h_j + dist_squared (size 1)
        message_dim = emb_dim * 2 + 1
        coord_input_dim = hidden_dim 
        # Coordinate updates are a scalar force applied along the edge being used to build the edge message
        coord_update_dim = 1 
        # h_n + aggregated message functions (size hidden_dim)
        feature_input_dim = emb_dim + hidden_dim 

        self.msg_mlp = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            activation_func,
            nn.Linear(hidden_dim, hidden_dim),
            activation_func
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_input_dim, hidden_dim),
            activation_func,
            nn.Linear(hidden_dim, coord_update_dim)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(feature_input_dim, hidden_dim),
            activation_func,
            nn.Linear(hidden_dim, emb_dim)
        )

    def coord2radial(self, edge_index, coord):
        """Radial is squared Euclidean distance, fyi"""
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        return radial, coord_diff/torch.sqrt(radial + 1e-8)

    def forward(self, x, h, edge_index):
        """
        Forward pass
        
        :param h: Scalar features
        :param x: Position (Vector) features (Equivariant over these)
        :param edge_index: Edge indices for all edges in the graph: [[n1, n2] x # edges]
        """

        n1, n2 = edge_index

        squared_dist, norm_coord_diff = self.coord2radial(edge_index, x)
        msgs = self.msg_mlp(torch.cat([h[n1], h[n2], squared_dist],dim=1))

        # Update x based on aggregated coordinate updates
        coord_updates = norm_coord_diff * self.coord_mlp(msgs)
        x_new = x + torch_scatter.scatter(coord_updates, n1, dim=0, reduce='sum', dim_size=x.size(0))
        
        feature_updates = torch_scatter.scatter(msgs, n1, dim=0, reduce='sum', dim_size=h.size(0))
        h_new = h + self.node_mlp(torch.cat([feature_updates, h], dim=1))

        return x_new, h_new

class SquaresModel(nn.Module):
    def __init__(self, n_nodes=4, n_layers=4, emb_dim = 64):
        super().__init__()
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.n_nodes = n_nodes

        # Bidirectional edge indices
        self.edges = torch.Tensor([[a, b,] for a in range(n_nodes) for b in range(n_nodes) if a !=b ])
        
        self.egnn_trunk = nn.ModuleList([E2_EGNN(hidden_dim=emb_dim,emb_dim=emb_dim) for _ in range(n_layers)])

    def get_batched_edges(self, batch_size):
        num_edges = self.edges.size(0)
        batches = []

        # Add the edge connections for all of the indices in the next graph in the batch. It should just be the previous node indices + n_nodes.
        for i in range(batch_size):
            batches += self.edges.clone() + i*self.num_nodes

        return torch.cat(batches, dim=0)   

    def forward(self, x):

        batch_size = x.size(0)
        x_flat = x.view(-1,2)

        edge_index = self.get_batched_edges(batch_size)

        # Initial hidden state embeddings
        h = torch.ones(x_flat.size(0),self.emb_dim, device = x.device)
        for layer in self.egnn_trunk:
            x_flat, h = layer(x, h, self.edge_index)
        return x_flat.view(-1, 4, 2)
    
class SquareModelTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    def get_loss(self, batch):
        """
        Loss function for my Square-Generating model
        
        :param batch: Batched Quadrilateral input data.
        """
        # cdist returns (Batch, 4, 4) distance matrix
        dists_matrix = torch.cdist(batch, batch)
        
        # Extract unique edges. For 4 nodes, upper triangle indices:
        # (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        unique_edges = torch.triu_indices(4, 4, offset=1)
        edge_lengths = dists_matrix[:, unique_edges[0], unique_edges[1]] # (Batch, 6)
        
        # Calculate stats per graph
        mu = edge_lengths.mean(dim=1)     # (Batch,)
        var = edge_lengths.var(dim=1, unbiased=False) # (Batch,)
        
        # Loss per graph
        loss_per_graph = (var / (mu**2 + 1e-6)) + torch.log(mu)**2
        
        return loss_per_graph.mean()
    
    def train_step(self, batch):
        """A single step in the training process"""

        self.optimizer.zero_grad()
        x_pred = self.model(batch)
        loss = self.get_loss(x_pred)
        loss.backward()

        self.optimizer.step()
        return loss.item(), x_pred
    

def plot_quadrilaterals_dense(data, num_to_plot=20, title="Squares"):
    """
    Plots a dense graph (fully connected) for each set of points.
    """
    points = data.reshape(-1, 4, 2) if data.ndim == 2 else data
    limit = min(len(points), num_to_plot)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Color configuration
    lc = ['00F','FF0','0FF','00F']
    vertex0_color = '#FFD700' # Gold
    
    for i in range(limit):
        sq = points[i]
        
        for p1_idx, p2_idx in itertools.combinations(range(4), 2):
            p1 = sq[p1_idx]
            p2 = sq[p2_idx]
            
            ax.plot(
                [p1[0], p2[0]], 
                [p1[1], p2[1]], 
                color= f"#{lc[p1_idx]}{lc[p2_idx]}", 
                linewidth=1.5, 
                alpha=0.6  # Lower alpha to handle the overlap gracefully
            )

        # Mark Vertex 0 for reference
        ax.scatter(sq[0, 0], sq[0, 1], color=vertex0_color, edgecolors='black', s=50, zorder=10)

    # Simplified Legend
    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=vertex0_color, markersize=8, markeredgecolor='k', label='Vertex 0 (Start)'),
    ]

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_title(title)
    ax.legend(handles=legend_elements, loc='upper center')
    
    plt.show()

def plot_input_and_output_quads(input_quads: np.ndarray, output_quads: np.ndarray, title="Squarification"):
    assert input_quads.shape == output_quads.shape
    n = input_quads.shape[0]
    fig, ax = plt.subplots(figsize=(8,8))

    lc = ['00F','FF0','0FF','00F']
    v_color = '#FFD700'

    # Plot the inputs in Red
    for i in range(n):
        q = input_quads[i]

        for p1_idx, p2_idx in itertools.combinations(range(4),2):
            p1, p2 = q[p1_idx], q[p2_idx]

            ax.plot(
                [p1[0],p2[0]],
                [p1[1],p2[1]],
                color='red',
                linewidth=1.5,
                alpha=0.4 # Keep the originals relatively more translucent, like ghosts.
            )
        
        ax.scatter(q[0,0], q[0,1], color='gray', edgecolors=None, s=25, zorder=10)
    
    for i in range(n):
        sq = output_quads[i]
        for p1_idx, p2_idx in itertools.combinations(range(4), 2):
            p1, p2 = sq[p1_idx], sq[p2_idx]

            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color = f"#{lc[p1_idx]}{lc[p2_idx]}",
                linewidth=1.5,
                alpha = 0.7
            )
        
        ax.scatter(sq[0, 0], sq[0, 1], color=v_color, edgecolors='black', s=50, zorder=10)
    
    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=v_color, markersize=8, markeredgecolor='k', label='Model Output'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=4, markeredgecolor=None, label='Input Shape'),
    ]

    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.legend(handles=legend_elements, loc='upper center')