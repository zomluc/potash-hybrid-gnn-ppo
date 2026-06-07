from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn
from torch.distributions import Normal


@dataclass
class ActorOutput:
    mean: Tensor
    log_std: Tensor


class FiLMConditioner(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scale = nn.Linear(hidden_dim, hidden_dim)
        self.shift = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, features: Tensor, context: Tensor) -> Tensor:
        scale = torch.tanh(self.scale(context))
        shift = self.shift(context)
        return features * (1.0 + scale) + shift


class MaskedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, features: Tensor, query: Tensor, mask: Tensor | None = None) -> Tensor:
        if features.size(0) == 0:
            return torch.zeros_like(query)
        query_hidden = self.query_proj(query)
        keys = self.key_proj(features)
        values = self.value_proj(features)
        scores = torch.matmul(keys, query_hidden) * self.scale
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.to(dtype=torch.bool)
            if not torch.any(mask):
                return torch.zeros_like(query)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=0)
        return torch.sum(attention.unsqueeze(-1) * values, dim=0)


def masked_mean_pool(features: Tensor, mask: Tensor | None = None) -> Tensor:
    if features.size(0) == 0:
        raise ValueError("masked_mean_pool requires a non-empty feature tensor.")
    if mask is None:
        return features.mean(dim=0)
    if mask.dtype != torch.bool:
        mask = mask.to(dtype=torch.bool)
    if not torch.any(mask):
        return torch.zeros(features.size(-1), device=features.device, dtype=features.dtype)
    return features[mask].mean(dim=0)


class TemporalContextEncoder(nn.Module):
    def __init__(self, global_dim: int, hidden_dim: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, global_feature_history: Tensor) -> Tensor:
        if global_feature_history.dim() == 2:
            global_feature_history = global_feature_history.unsqueeze(0)
        projected = self.input_proj(global_feature_history)
        _, hidden = self.gru(projected)
        temporal_context = hidden[-1]
        return temporal_context.squeeze(0)


class RiskConditionedMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0, use_risk_film: bool = True, use_edge_enhancement: bool = True):
        super().__init__()
        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.msg_linear = nn.Linear(hidden_dim * 3, hidden_dim)
        self.gate_linear = nn.Linear(hidden_dim * 4, hidden_dim)
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.film = FiLMConditioner(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_risk_film = use_risk_film
        self.use_edge_enhancement = use_edge_enhancement

    def forward(self, node_features: Tensor, edge_index: Tensor, edge_features: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        src_index = edge_index[0].long()
        dst_index = edge_index[1].long()

        src_features = node_features[src_index]
        dst_features = node_features[dst_index]
        repeated_context = context.unsqueeze(0).expand(edge_features.size(0), -1)

        if self.use_edge_enhancement:
            edge_inputs = torch.cat([src_features, dst_features, edge_features, repeated_context], dim=-1)
            updated_edge = self.edge_norm(edge_features + self.dropout(self.edge_update(edge_inputs)))
        else:
            updated_edge = edge_features

        gate_inputs = torch.cat([src_features, dst_features, updated_edge, repeated_context], dim=-1)
        gate = torch.sigmoid(self.gate_linear(gate_inputs))
        messages = self.msg_linear(torch.cat([src_features, updated_edge, repeated_context], dim=-1)) * gate

        aggregated = torch.zeros(
            node_features.size(0),
            messages.size(-1),
            device=node_features.device,
            dtype=messages.dtype,
        )
        aggregated.index_add_(0, dst_index, messages)

        degree = torch.zeros(node_features.size(0), 1, device=node_features.device, dtype=messages.dtype)
        degree.index_add_(0, dst_index, torch.ones(dst_index.size(0), 1, device=node_features.device, dtype=messages.dtype))
        aggregated = aggregated / degree.clamp_min(1.0)

        self_term = self.self_linear(node_features)
        hidden = torch.cat([self_term, aggregated], dim=-1)
        hidden = self.out_linear(hidden)
        if self.use_risk_film:
            hidden = self.film(hidden, context)
        hidden = self.dropout(hidden)
        hidden = torch.relu(self.norm(node_features + hidden))
        return hidden, updated_edge


class GraphEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_temporal_history: bool = True,
        use_attention_pooling: bool = True,
        use_risk_film: bool = True,
        use_edge_enhancement: bool = True,
    ):
        super().__init__()
        self.node_input = nn.Linear(node_dim, hidden_dim)
        self.edge_input = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                RiskConditionedMessageLayer(
                    hidden_dim,
                    dropout=dropout,
                    use_risk_film=use_risk_film,
                    use_edge_enhancement=use_edge_enhancement,
                )
                for _ in range(num_layers)
            ]
        )
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal_encoder = TemporalContextEncoder(global_dim=global_dim, hidden_dim=hidden_dim)
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.supplier_pool = MaskedAttentionPooling(hidden_dim)
        self.port_pool = MaskedAttentionPooling(hidden_dim)
        self.route_pool = MaskedAttentionPooling(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.use_temporal_history = use_temporal_history
        self.use_attention_pooling = use_attention_pooling
        self.use_edge_enhancement = use_edge_enhancement

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        global_features: Tensor,
        global_feature_history: Tensor,
        focal_node_index: int = 9,
    ) -> Dict[str, Tensor]:
        supplier_mask = node_features[:, 0] > 0.5
        port_mask = node_features[:, 1] > 0.5
        hidden = torch.relu(self.node_input(node_features))
        edge_hidden = torch.relu(self.edge_input(edge_features))
        global_embedding = self.global_proj(global_features)
        temporal_embedding = self.temporal_encoder(global_feature_history) if self.use_temporal_history else torch.zeros_like(global_embedding)
        context = self.context_fusion(torch.cat([global_embedding, temporal_embedding], dim=-1))

        for layer in self.layers:
            hidden, edge_hidden = layer(hidden, edge_index, edge_hidden, context)

        focal_embedding = hidden[focal_node_index]
        query = self.query_proj(torch.cat([focal_embedding, context], dim=-1))
        if self.use_attention_pooling:
            supplier_embedding = self.supplier_pool(hidden, query, supplier_mask)
            port_embedding = self.port_pool(hidden, query, port_mask)
            route_embedding = self.route_pool(edge_hidden, query) if self.use_edge_enhancement else torch.zeros_like(focal_embedding)
        else:
            supplier_embedding = masked_mean_pool(hidden, supplier_mask)
            port_embedding = masked_mean_pool(hidden, port_mask)
            route_embedding = masked_mean_pool(edge_hidden) if self.use_edge_enhancement else torch.zeros_like(focal_embedding)
        joint_embedding = self.readout(
            torch.cat([focal_embedding, supplier_embedding, port_embedding, route_embedding, context], dim=-1)
        )
        return {
            "joint_embedding": joint_embedding,
            "route_embedding": route_embedding,
            "context_embedding": context,
        }


class GATMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.node_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        src_index = edge_index[0].long()
        dst_index = edge_index[1].long()
        projected = self.node_proj(node_features).view(node_features.size(0), self.num_heads, self.head_dim)
        src_features = projected[src_index]
        dst_features = projected[dst_index]

        scores = (src_features * self.attn_src.unsqueeze(0)).sum(dim=-1)
        scores = scores + (dst_features * self.attn_dst.unsqueeze(0)).sum(dim=-1)
        scores = self.leaky_relu(scores)

        attention = torch.zeros_like(scores)
        for node_id in torch.unique(dst_index):
            mask = dst_index == node_id
            attention[mask] = torch.softmax(scores[mask], dim=0)

        messages = src_features * attention.unsqueeze(-1)
        aggregated = torch.zeros(
            node_features.size(0),
            self.num_heads,
            self.head_dim,
            device=node_features.device,
            dtype=node_features.dtype,
        )
        aggregated.index_add_(0, dst_index, messages)
        aggregated = aggregated.reshape(node_features.size(0), -1)

        hidden = self.out_proj(aggregated)
        hidden = self.dropout(hidden)
        return torch.relu(self.norm(node_features + hidden))


class GATGraphEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        global_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_heads: int = 4,
        use_temporal_history: bool = True,
        use_attention_pooling: bool = True,
    ):
        super().__init__()
        self.node_input = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GATMessageLayer(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal_encoder = TemporalContextEncoder(global_dim=global_dim, hidden_dim=hidden_dim)
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.supplier_pool = MaskedAttentionPooling(hidden_dim)
        self.port_pool = MaskedAttentionPooling(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.use_temporal_history = use_temporal_history
        self.use_attention_pooling = use_attention_pooling

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        global_features: Tensor,
        global_feature_history: Tensor,
        focal_node_index: int = 9,
    ) -> Dict[str, Tensor]:
        supplier_mask = node_features[:, 0] > 0.5
        port_mask = node_features[:, 1] > 0.5
        hidden = torch.relu(self.node_input(node_features))
        global_embedding = self.global_proj(global_features)
        temporal_embedding = self.temporal_encoder(global_feature_history) if self.use_temporal_history else torch.zeros_like(global_embedding)
        context = self.context_fusion(torch.cat([global_embedding, temporal_embedding], dim=-1))

        for layer in self.layers:
            hidden = layer(hidden, edge_index)

        focal_embedding = hidden[focal_node_index]
        query = self.query_proj(torch.cat([focal_embedding, context], dim=-1))
        if self.use_attention_pooling:
            supplier_embedding = self.supplier_pool(hidden, query, supplier_mask)
            port_embedding = self.port_pool(hidden, query, port_mask)
        else:
            supplier_embedding = masked_mean_pool(hidden, supplier_mask)
            port_embedding = masked_mean_pool(hidden, port_mask)
        joint_embedding = self.readout(torch.cat([focal_embedding, supplier_embedding, port_embedding, context], dim=-1))
        return {
            "joint_embedding": joint_embedding,
            "context_embedding": context,
        }


class GNNActor(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        init_log_std: float = -0.5,
        use_temporal_history: bool = True,
        use_attention_pooling: bool = True,
        use_risk_film: bool = True,
        use_edge_enhancement: bool = True,
    ):
        super().__init__()
        self.encoder = GraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_temporal_history=use_temporal_history,
            use_attention_pooling=use_attention_pooling,
            use_risk_film=use_risk_film,
            use_edge_enhancement=use_edge_enhancement,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

    def encode(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        global_features: Tensor,
        global_feature_history: Tensor,
    ) -> Tensor:
        encoded = self.encoder(node_features, edge_index, edge_features, global_features, global_feature_history)
        return self.policy_head(encoded["joint_embedding"])

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        global_features: Tensor,
        global_feature_history: Tensor,
    ) -> ActorOutput:
        hidden = self.encode(node_features, edge_index, edge_features, global_features, global_feature_history)
        mean = self.mean_head(hidden)
        log_std = self.log_std.expand_as(mean)
        return ActorOutput(mean=mean, log_std=log_std)

    def distribution(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        global_features: Tensor,
        global_feature_history: Tensor,
    ) -> Normal:
        output = self.forward(node_features, edge_index, edge_features, global_features, global_feature_history)
        return Normal(output.mean, output.log_std.exp())

    def sample_action(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        global_features: Tensor,
        global_feature_history: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(node_features, edge_index, edge_features, global_features, global_feature_history)
        if deterministic:
            action = distribution.mean
        else:
            action = distribution.rsample()
        log_prob = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return action, log_prob, entropy


class GNNCritic(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_temporal_history: bool = True,
        use_attention_pooling: bool = True,
        use_risk_film: bool = True,
        use_edge_enhancement: bool = True,
    ):
        super().__init__()
        self.encoder = GraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_temporal_history=use_temporal_history,
            use_attention_pooling=use_attention_pooling,
            use_risk_film=use_risk_film,
            use_edge_enhancement=use_edge_enhancement,
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        global_features: Tensor,
        global_feature_history: Tensor,
    ) -> Tensor:
        encoded = self.encoder(
            node_features,
            edge_index,
            edge_features,
            global_features,
            global_feature_history,
        )
        return self.value_head(encoded["joint_embedding"]).squeeze(-1)


class GATActor(nn.Module):
    def __init__(
        self,
        node_dim: int,
        global_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        init_log_std: float = -0.5,
        use_temporal_history: bool = True,
        use_attention_pooling: bool = True,
    ):
        super().__init__()
        self.encoder = GATGraphEncoder(
            node_dim=node_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_temporal_history=use_temporal_history,
            use_attention_pooling=use_attention_pooling,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = StructuredActionHead(hidden_dim=hidden_dim, action_dim=action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

    def encode(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        global_features: Tensor,
        global_feature_history: Tensor,
    ) -> Tensor:
        encoded = self.encoder(node_features, edge_index, global_features, global_feature_history)
        return self.policy_head(encoded["joint_embedding"])

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        global_features: Tensor,
        global_feature_history: Tensor,
    ) -> ActorOutput:
        hidden = self.encode(node_features, edge_index, global_features, global_feature_history)
        mean = self.mean_head(hidden)
        log_std = self.log_std.expand_as(mean)
        return ActorOutput(mean=mean, log_std=log_std)

    def distribution(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        global_features: Tensor,
        global_feature_history: Tensor,
    ) -> Normal:
        output = self.forward(node_features, edge_index, global_features, global_feature_history)
        return Normal(output.mean, output.log_std.exp())

    def sample_action(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        global_features: Tensor,
        global_feature_history: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(node_features, edge_index, global_features, global_feature_history)
        if deterministic:
            action = distribution.mean
        else:
            action = distribution.rsample()
        log_prob = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return action, log_prob, entropy


class GATCritic(nn.Module):
    def __init__(
        self,
        node_dim: int,
        global_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_temporal_history: bool = True,
        use_attention_pooling: bool = True,
    ):
        super().__init__()
        self.encoder = GATGraphEncoder(
            node_dim=node_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_temporal_history=use_temporal_history,
            use_attention_pooling=use_attention_pooling,
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        global_features: Tensor,
        global_feature_history: Tensor,
    ) -> Tensor:
        encoded = self.encoder(node_features, edge_index, global_features, global_feature_history)
        return self.value_head(encoded["joint_embedding"]).squeeze(-1)


class MLPActor(nn.Module):
    def __init__(
        self,
        flat_obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(flat_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

    def encode(self, flat_observation: Tensor) -> Tensor:
        return self.policy_head(flat_observation)

    def forward(self, flat_observation: Tensor) -> ActorOutput:
        hidden = self.encode(flat_observation)
        mean = self.mean_head(hidden)
        log_std = self.log_std.expand_as(mean)
        return ActorOutput(mean=mean, log_std=log_std)

    def distribution(self, flat_observation: Tensor) -> Normal:
        output = self.forward(flat_observation)
        return Normal(output.mean, output.log_std.exp())

    def sample_action(self, flat_observation: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(flat_observation)
        if deterministic:
            action = distribution.mean
        else:
            action = distribution.rsample()
        log_prob = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return action, log_prob, entropy


class FlatFeatureEncoder(nn.Module):
    def __init__(self, flat_obs_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(flat_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, flat_observation: Tensor) -> Tensor:
        return self.net(flat_observation)


class HybridFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, graph_embedding: Tensor, flat_embedding: Tensor) -> Tensor:
        gate = self.gate(torch.cat([graph_embedding, flat_embedding], dim=-1))
        blended = gate * graph_embedding + (1.0 - gate) * flat_embedding
        return self.out(torch.cat([blended, flat_embedding], dim=-1))


class StructuredActionHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        if action_dim != 9:
            raise ValueError(f"StructuredActionHead expects action_dim=9, got {action_dim}.")
        self.supplier_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.control_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        supplier_logits = self.supplier_head(hidden)
        control_means = self.control_head(hidden)
        return torch.cat([supplier_logits, control_means], dim=-1)


class HybridActor(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        flat_obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        init_log_std: float = -0.5,
        use_temporal_history: bool = True,
        use_attention_pooling: bool = True,
        use_risk_film: bool = True,
        use_edge_enhancement: bool = True,
    ):
        super().__init__()
        self.graph_encoder = GraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_temporal_history=use_temporal_history,
            use_attention_pooling=use_attention_pooling,
            use_risk_film=use_risk_film,
            use_edge_enhancement=use_edge_enhancement,
        )
        self.flat_encoder = FlatFeatureEncoder(flat_obs_dim=flat_obs_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.fusion = HybridFusion(hidden_dim=hidden_dim, dropout=dropout)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = StructuredActionHead(hidden_dim=hidden_dim, action_dim=action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

    def encode(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        global_features: Tensor,
        global_feature_history: Tensor,
        flat_observation: Tensor,
    ) -> Tensor:
        graph_outputs = self.graph_encoder(
            node_features,
            edge_index,
            edge_features,
            global_features,
            global_feature_history,
        )
        flat_encoded = self.flat_encoder(flat_observation)
        fused = self.fusion(graph_outputs["joint_embedding"], flat_encoded)
        return self.policy_head(fused)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        global_features: Tensor,
        global_feature_history: Tensor,
        flat_observation: Tensor,
    ) -> ActorOutput:
        hidden = self.encode(
            node_features,
            edge_index,
            edge_features,
            global_features,
            global_feature_history,
            flat_observation,
        )
        mean = self.mean_head(hidden)
        log_std = self.log_std.expand_as(mean)
        return ActorOutput(mean=mean, log_std=log_std)

    def distribution(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        global_features: Tensor,
        global_feature_history: Tensor,
        flat_observation: Tensor,
    ) -> Normal:
        output = self.forward(
            node_features,
            edge_index,
            edge_features,
            global_features,
            global_feature_history,
            flat_observation,
        )
        return Normal(output.mean, output.log_std.exp())

    def sample_action(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        global_features: Tensor,
        global_feature_history: Tensor,
        flat_observation: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(
            node_features,
            edge_index,
            edge_features,
            global_features,
            global_feature_history,
            flat_observation,
        )
        if deterministic:
            action = distribution.mean
        else:
            action = distribution.rsample()
        log_prob = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return action, log_prob, entropy


class HybridCritic(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        flat_obs_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_temporal_history: bool = True,
        use_attention_pooling: bool = True,
        use_risk_film: bool = True,
        use_edge_enhancement: bool = True,
    ):
        super().__init__()
        self.graph_encoder = GraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_temporal_history=use_temporal_history,
            use_attention_pooling=use_attention_pooling,
            use_risk_film=use_risk_film,
            use_edge_enhancement=use_edge_enhancement,
        )
        self.flat_encoder = FlatFeatureEncoder(flat_obs_dim=flat_obs_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.fusion = HybridFusion(hidden_dim=hidden_dim, dropout=dropout)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        global_features: Tensor,
        global_feature_history: Tensor,
        flat_observation: Tensor,
    ) -> Tensor:
        graph_outputs = self.graph_encoder(
            node_features,
            edge_index,
            edge_features,
            global_features,
            global_feature_history,
        )
        flat_encoded = self.flat_encoder(flat_observation)
        fused = self.fusion(graph_outputs["joint_embedding"], flat_encoded)
        return self.value_head(fused).squeeze(-1)


class MLPCritic(nn.Module):
    def __init__(self, flat_obs_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(flat_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, flat_observation: Tensor) -> Tensor:
        return self.value_net(flat_observation).squeeze(-1)


class TransformerStateEncoder(nn.Module):
    def __init__(
        self,
        global_dim: int,
        flat_obs_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        num_layers: int = 1,
        num_heads: int = 4,
        max_tokens: int = 8,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")
        self.flat_proj = nn.Sequential(
            nn.Linear(flat_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.history_proj = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, max_tokens, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, global_features: Tensor, global_feature_history: Tensor, flat_observation: Tensor) -> Tensor:
        squeeze_batch = False
        if flat_observation.dim() == 1:
            flat_observation = flat_observation.unsqueeze(0)
            global_features = global_features.unsqueeze(0)
            global_feature_history = global_feature_history.unsqueeze(0)
            squeeze_batch = True
        flat_token = self.flat_proj(flat_observation).unsqueeze(1)
        global_token = self.global_proj(global_features).unsqueeze(1)
        history_tokens = self.history_proj(global_feature_history)
        cls_token = self.cls_token.expand(flat_observation.size(0), -1, -1)
        tokens = torch.cat([cls_token, flat_token, global_token, history_tokens], dim=1)
        if tokens.size(1) > self.position_embedding.size(1):
            raise ValueError(
                f"TransformerStateEncoder received {tokens.size(1)} tokens, but max_tokens={self.position_embedding.size(1)}."
            )
        tokens = tokens + self.position_embedding[:, : tokens.size(1), :]
        encoded = self.encoder(tokens)
        pooled = self.output_norm(encoded[:, 0, :])
        return pooled.squeeze(0) if squeeze_batch else pooled


class TransformerActor(nn.Module):
    def __init__(
        self,
        global_dim: int,
        flat_obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.encoder = TransformerStateEncoder(
            global_dim=global_dim,
            flat_obs_dim=flat_obs_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = StructuredActionHead(hidden_dim=hidden_dim, action_dim=action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

    def encode(self, global_features: Tensor, global_feature_history: Tensor, flat_observation: Tensor) -> Tensor:
        return self.policy_head(self.encoder(global_features, global_feature_history, flat_observation))

    def forward(self, global_features: Tensor, global_feature_history: Tensor, flat_observation: Tensor) -> ActorOutput:
        hidden = self.encode(global_features, global_feature_history, flat_observation)
        mean = self.mean_head(hidden)
        log_std = self.log_std.expand_as(mean)
        return ActorOutput(mean=mean, log_std=log_std)

    def distribution(self, global_features: Tensor, global_feature_history: Tensor, flat_observation: Tensor) -> Normal:
        output = self.forward(global_features, global_feature_history, flat_observation)
        return Normal(output.mean, output.log_std.exp())

    def sample_action(
        self,
        global_features: Tensor,
        global_feature_history: Tensor,
        flat_observation: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(global_features, global_feature_history, flat_observation)
        if deterministic:
            action = distribution.mean
        else:
            action = distribution.rsample()
        log_prob = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return action, log_prob, entropy


class TransformerCritic(nn.Module):
    def __init__(
        self,
        global_dim: int,
        flat_obs_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TransformerStateEncoder(
            global_dim=global_dim,
            flat_obs_dim=flat_obs_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_features: Tensor, global_feature_history: Tensor, flat_observation: Tensor) -> Tensor:
        hidden = self.encoder(global_features, global_feature_history, flat_observation)
        return self.value_head(hidden).squeeze(-1)


class ActorCriticNetworks(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        flat_obs_dim: int,
        action_dim: int,
        model_type: str = "gnn",
        actor_hidden_dim: int = 128,
        critic_hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        dropout: float = 0.1,
        actor_init_log_std: float = -0.5,
        gnn_use_temporal_history: bool = True,
        gnn_use_attention_pooling: bool = True,
        gnn_use_risk_film: bool = True,
        gnn_use_edge_enhancement: bool = True,
    ):
        super().__init__()
        if model_type not in {"gnn", "gat", "mlp", "hybrid", "transformer"}:
            raise ValueError(f"Unsupported model_type: {model_type}")
        self.model_type = model_type
        if self.model_type == "gnn":
            self.actor = GNNActor(
                node_dim=node_dim,
                edge_dim=edge_dim,
                global_dim=global_dim,
                action_dim=action_dim,
                hidden_dim=actor_hidden_dim,
                num_layers=num_gnn_layers,
                dropout=dropout,
                init_log_std=actor_init_log_std,
                use_temporal_history=gnn_use_temporal_history,
                use_attention_pooling=gnn_use_attention_pooling,
                use_risk_film=gnn_use_risk_film,
                use_edge_enhancement=gnn_use_edge_enhancement,
            )
        elif self.model_type == "gat":
            self.actor = GATActor(
                node_dim=node_dim,
                global_dim=global_dim,
                action_dim=action_dim,
                hidden_dim=actor_hidden_dim,
                num_layers=num_gnn_layers,
                dropout=dropout,
                init_log_std=actor_init_log_std,
                use_temporal_history=gnn_use_temporal_history,
                use_attention_pooling=gnn_use_attention_pooling,
            )
        elif self.model_type == "transformer":
            self.actor = TransformerActor(
                global_dim=global_dim,
                flat_obs_dim=flat_obs_dim,
                action_dim=action_dim,
                hidden_dim=actor_hidden_dim,
                dropout=dropout,
                init_log_std=actor_init_log_std,
            )
        elif self.model_type == "hybrid":
            self.actor = HybridActor(
                node_dim=node_dim,
                edge_dim=edge_dim,
                global_dim=global_dim,
                flat_obs_dim=flat_obs_dim,
                action_dim=action_dim,
                hidden_dim=actor_hidden_dim,
                num_layers=num_gnn_layers,
                dropout=dropout,
                init_log_std=actor_init_log_std,
                use_temporal_history=gnn_use_temporal_history,
                use_attention_pooling=gnn_use_attention_pooling,
                use_risk_film=gnn_use_risk_film,
                use_edge_enhancement=gnn_use_edge_enhancement,
            )
        else:
            self.actor = MLPActor(
                flat_obs_dim=flat_obs_dim,
                action_dim=action_dim,
                hidden_dim=actor_hidden_dim,
                dropout=dropout,
                init_log_std=actor_init_log_std,
            )
        if self.model_type == "gnn":
            self.critic = GNNCritic(
                node_dim=node_dim,
                edge_dim=edge_dim,
                global_dim=global_dim,
                hidden_dim=critic_hidden_dim,
                num_layers=num_gnn_layers,
                dropout=dropout,
                use_temporal_history=gnn_use_temporal_history,
                use_attention_pooling=gnn_use_attention_pooling,
                use_risk_film=gnn_use_risk_film,
                use_edge_enhancement=gnn_use_edge_enhancement,
            )
        elif self.model_type == "gat":
            self.critic = GATCritic(
                node_dim=node_dim,
                global_dim=global_dim,
                hidden_dim=critic_hidden_dim,
                num_layers=num_gnn_layers,
                dropout=dropout,
                use_temporal_history=gnn_use_temporal_history,
                use_attention_pooling=gnn_use_attention_pooling,
            )
        elif self.model_type == "hybrid":
            self.critic = HybridCritic(
                node_dim=node_dim,
                edge_dim=edge_dim,
                global_dim=global_dim,
                flat_obs_dim=flat_obs_dim,
                hidden_dim=critic_hidden_dim,
                num_layers=num_gnn_layers,
                dropout=dropout,
                use_temporal_history=gnn_use_temporal_history,
                use_attention_pooling=gnn_use_attention_pooling,
                use_risk_film=gnn_use_risk_film,
                use_edge_enhancement=gnn_use_edge_enhancement,
            )
        elif self.model_type == "transformer":
            self.critic = TransformerCritic(
                global_dim=global_dim,
                flat_obs_dim=flat_obs_dim,
                hidden_dim=critic_hidden_dim,
                dropout=dropout,
            )
        else:
            self.critic = MLPCritic(flat_obs_dim=flat_obs_dim, hidden_dim=critic_hidden_dim, dropout=dropout)

    def distribution(self, observation: Dict[str, Tensor]) -> Normal:
        if self.model_type == "gnn":
            return self.actor.distribution(
                node_features=observation["node_features"],
                edge_index=observation["edge_index"],
                edge_features=observation["edge_features"],
                global_features=observation["global_features"],
                global_feature_history=observation["global_feature_history"],
            )
        if self.model_type == "gat":
            return self.actor.distribution(
                node_features=observation["node_features"],
                edge_index=observation["edge_index"],
                global_features=observation["global_features"],
                global_feature_history=observation["global_feature_history"],
            )
        if self.model_type == "hybrid":
            return self.actor.distribution(
                node_features=observation["node_features"],
                edge_index=observation["edge_index"],
                edge_features=observation["edge_features"],
                global_features=observation["global_features"],
                global_feature_history=observation["global_feature_history"],
                flat_observation=observation["flat_observation"],
            )
        if self.model_type == "transformer":
            return self.actor.distribution(
                global_features=observation["global_features"],
                global_feature_history=observation["global_feature_history"],
                flat_observation=observation["flat_observation"],
            )
        return self.actor.distribution(observation["flat_observation"])

    def act(self, observation: Dict[str, Tensor], deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        if self.model_type == "gnn":
            return self.actor.sample_action(
                node_features=observation["node_features"],
                edge_index=observation["edge_index"],
                edge_features=observation["edge_features"],
                global_features=observation["global_features"],
                global_feature_history=observation["global_feature_history"],
                deterministic=deterministic,
            )
        if self.model_type == "gat":
            return self.actor.sample_action(
                node_features=observation["node_features"],
                edge_index=observation["edge_index"],
                global_features=observation["global_features"],
                global_feature_history=observation["global_feature_history"],
                deterministic=deterministic,
            )
        if self.model_type == "hybrid":
            return self.actor.sample_action(
                node_features=observation["node_features"],
                edge_index=observation["edge_index"],
                edge_features=observation["edge_features"],
                global_features=observation["global_features"],
                global_feature_history=observation["global_feature_history"],
                flat_observation=observation["flat_observation"],
                deterministic=deterministic,
            )
        if self.model_type == "transformer":
            return self.actor.sample_action(
                global_features=observation["global_features"],
                global_feature_history=observation["global_feature_history"],
                flat_observation=observation["flat_observation"],
                deterministic=deterministic,
            )
        return self.actor.sample_action(observation["flat_observation"], deterministic=deterministic)

    def value(self, observation: Dict[str, Tensor]) -> Tensor:
        if self.model_type == "gnn":
            return self.critic(
                observation["node_features"],
                observation["edge_index"],
                observation["edge_features"],
                observation["global_features"],
                observation["global_feature_history"],
            )
        if self.model_type == "gat":
            return self.critic(
                observation["node_features"],
                observation["edge_index"],
                observation["global_features"],
                observation["global_feature_history"],
            )
        if self.model_type == "hybrid":
            return self.critic(
                observation["node_features"],
                observation["edge_index"],
                observation["edge_features"],
                observation["global_features"],
                observation["global_feature_history"],
                observation["flat_observation"],
            )
        if self.model_type == "transformer":
            return self.critic(
                observation["global_features"],
                observation["global_feature_history"],
                observation["flat_observation"],
            )
        return self.critic(observation["flat_observation"])


def observation_to_torch(observation: Dict[str, Tensor | torch.Tensor | object], device: torch.device | str = "cpu") -> Dict[str, Tensor]:
    tensors: Dict[str, Tensor] = {}
    for key, value in observation.items():
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if key == "edge_index":
            tensors[key] = tensor.to(device=device, dtype=torch.long)
        else:
            tensors[key] = tensor.to(device=device, dtype=torch.float32)
    return tensors
