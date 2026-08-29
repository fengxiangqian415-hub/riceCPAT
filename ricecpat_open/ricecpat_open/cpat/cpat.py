import sys
import os

# Add the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from tools.encoder import Encoder
from tools.positional_encoding import generate_original_PE, generate_regular_PE


class StageEncoder(nn.Module):
    """Stage encoder.

    Encodes a single temporal stage and maps the variable-length stage
    sequence to a fixed-dimensional embedding.
    """

    def __init__(self, d_model: int, stage_embed_dim: int, pooling_method: str = 'mean'):
        super().__init__()

        self.d_model = d_model
        self.stage_embed_dim = stage_embed_dim
        self.pooling_method = pooling_method

        # Intra-stage sequence encoder
        self.stage_encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=min(8, max(1, d_model // 64)),  # adaptive number of heads
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )

        # Stage embedding projection
        self.stage_projection = nn.Sequential(
            nn.Linear(d_model, stage_embed_dim),
            nn.LayerNorm(stage_embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(
        self,
        stage_sequence: torch.Tensor,
        stage_mask: torch.Tensor = None,
        apply_attention_mask: bool = False,
    ) -> torch.Tensor:
        """Embed a single temporal stage into a fixed dimension.

        Parameters
        ----------
        stage_sequence: torch.Tensor
            Stage sequence (batch_size, stage_len, d_model)
        stage_mask: torch.Tensor
            Stage mask (batch_size, stage_len)

        Returns
        -------
        stage_embedding: torch.Tensor
            Fixed-dimensional stage embedding (batch_size, stage_embed_dim)
        """
        batch_size, stage_len, d_model = stage_sequence.shape
        # stage_sequence shape: (batch_size, stage_len, d_model) e.g. (2, 4, 64)

        # Intra-stage encoding enhancement
        if stage_len > 1:
            if apply_attention_mask and stage_mask is not None:
                # After phenology-guided segmentation, different samples may
                # have different stage lengths, so the intra-stage
                # self-attention must mask padded positions. Rows that are
                # entirely padded are not fed to the Transformer to avoid NaNs
                # from an all-masked softmax.
                valid_rows = stage_mask.bool().any(dim=1)
                encoded_sequence = torch.zeros_like(stage_sequence)
                if valid_rows.any():
                    valid_encoded = self.stage_encoder(
                        stage_sequence[valid_rows],
                        src_key_padding_mask=~stage_mask[valid_rows].bool(),
                    )
                    encoded_sequence[valid_rows] = valid_encoded
            else:
                # The "equal" mode keeps the original computation path so that
                # the default inference behavior of legacy weights is unchanged.
                encoded_sequence = self.stage_encoder(stage_sequence)
            # encoded_sequence shape: (batch_size, stage_len, d_model) e.g. (2, 4, 64)
        else:
            encoded_sequence = stage_sequence
            # encoded_sequence shape: (batch_size, stage_len, d_model) e.g. (2, 4, 64)

        # Pooling: aggregate the variable-length stage into a fixed vector
        if self.pooling_method == 'mean':
            if stage_mask is not None:
                mask_expanded = stage_mask.unsqueeze(-1)
                # mask_expanded shape: (batch_size, stage_len, 1) e.g. (2, 4, 1)
                masked_sequence = encoded_sequence * mask_expanded
                # masked_sequence shape: (batch_size, stage_len, d_model) e.g. (2, 4, 64)
                seq_lengths = stage_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                # seq_lengths shape: (batch_size, 1) e.g. (2, 1)
                stage_vector = masked_sequence.sum(dim=1) / seq_lengths
                # stage_vector shape: (batch_size, d_model) e.g. (2, 64)
            else:
                stage_vector = encoded_sequence.mean(dim=1)
                # stage_vector shape: (batch_size, d_model) e.g. (2, 64)

        elif self.pooling_method == 'max':
            if stage_mask is not None:
                mask_expanded = stage_mask.unsqueeze(-1)
                # mask_expanded shape: (batch_size, stage_len, 1) e.g. (2, 4, 1)
                masked_sequence = encoded_sequence.masked_fill(mask_expanded == 0, -1e9)
                # masked_sequence shape: (batch_size, stage_len, d_model) e.g. (2, 4, 64)
                stage_vector, _ = torch.max(masked_sequence, dim=1)
                # stage_vector shape: (batch_size, d_model) e.g. (2, 64)
            else:
                stage_vector, _ = torch.max(encoded_sequence, dim=1)
                # stage_vector shape: (batch_size, d_model) e.g. (2, 64)

        elif self.pooling_method == 'last':
            if stage_mask is not None:
                last_indices = (stage_mask.sum(dim=1) - 1).long().clamp(min=0, max=stage_len-1)
                # last_indices shape: (batch_size,) e.g. (2,)
                stage_vector = encoded_sequence[torch.arange(batch_size), last_indices]
                # stage_vector shape: (batch_size, d_model) e.g. (2, 64)
            else:
                stage_vector = encoded_sequence[:, -1, :]
                # stage_vector shape: (batch_size, d_model) e.g. (2, 64)

        else:  # 'first' or default
            stage_vector = encoded_sequence[:, 0, :]
            # stage_vector shape: (batch_size, d_model) e.g. (2, 64)

        # Project to the fixed embedding space
        stage_embedding = self.stage_projection(stage_vector)
        # stage_vector input shape: (batch_size, d_model) e.g. (2, 64)
        # stage_embedding output shape: (batch_size, stage_embed_dim) e.g. (2, 64)

        if apply_attention_mask and stage_mask is not None:
            valid_stage = stage_mask.bool().any(dim=1, keepdim=True)
            stage_embedding = stage_embedding * valid_stage.to(stage_embedding.dtype)

        return stage_embedding


class StagedCrossAttention(nn.Module):
    """Cross-stage attention mechanism.

    Computes attention across the temporal stages in a permutation-invariant
    manner.
    """

    def __init__(self, stage_embed_dim: int, num_heads: int = 8, dropout: float = 0.3):
        super().__init__()

        self.stage_embed_dim = stage_embed_dim
        self.num_heads = num_heads
        self.head_dim = stage_embed_dim // num_heads

        assert stage_embed_dim % num_heads == 0, "stage_embed_dim must be divisible by num_heads"

        # Attention projections
        self.W_q = nn.Linear(stage_embed_dim, stage_embed_dim)
        self.W_k = nn.Linear(stage_embed_dim, stage_embed_dim)
        self.W_v = nn.Linear(stage_embed_dim, stage_embed_dim)
        self.W_o = nn.Linear(stage_embed_dim, stage_embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(stage_embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize the weights."""
        for layer in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        stage_embeddings: torch.Tensor,
        stage_valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute cross-attention in the stage embedding space.

        Parameters
        ----------
        stage_embeddings: torch.Tensor
            Embeddings of the temporal stages (batch_size, num_stages, stage_embed_dim)

        Returns
        -------
        output: torch.Tensor
            Cross-attention output (batch_size, num_stages, stage_embed_dim)
        """
        batch_size, num_stages, embed_dim = stage_embeddings.shape
        # stage_embeddings shape: (batch_size, num_stages, stage_embed_dim) e.g. (2, 3, 64)

        # Compute Q, K, V for all stages
        Q = self.W_q(stage_embeddings)
        # Q shape: (batch_size, num_stages, stage_embed_dim) e.g. (2, 3, 64)
        K = self.W_k(stage_embeddings)
        # K shape: (batch_size, num_stages, stage_embed_dim) e.g. (2, 3, 64)
        V = self.W_v(stage_embeddings)
        # V shape: (batch_size, num_stages, stage_embed_dim) e.g. (2, 3, 64)

        # Reshape into multi-head form
        Q = Q.view(batch_size, num_stages, self.num_heads, self.head_dim).transpose(1, 2)
        # Q shape: (batch_size, num_heads, num_stages, head_dim) e.g. (2, 8, 3, 8)
        K = K.view(batch_size, num_stages, self.num_heads, self.head_dim).transpose(1, 2)
        # K shape: (batch_size, num_heads, num_stages, head_dim) e.g. (2, 8, 3, 8)
        V = V.view(batch_size, num_stages, self.num_heads, self.head_dim).transpose(1, 2)
        # V shape: (batch_size, num_heads, num_stages, head_dim) e.g. (2, 8, 3, 8)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # K.transpose(-2, -1) shape: (batch_size, num_heads, head_dim, num_stages) e.g. (2, 8, 8, 3)
        # scores shape: (batch_size, num_heads, num_stages, num_stages) e.g. (2, 8, 3, 3)

        # Numerical stability check
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores = torch.zeros_like(scores)

        if stage_valid_mask is not None:
            if not stage_valid_mask.bool().any(dim=1).all():
                raise ValueError("Each sample must have at least one valid phenological stage.")
            invalid_keys = ~stage_valid_mask.bool()[:, None, None, :]
            scores = scores.masked_fill(invalid_keys, torch.finfo(scores.dtype).min)

        # Softmax normalization
        attention_weights = F.softmax(scores, dim=-1)
        # attention_weights shape: (batch_size, num_heads, num_stages, num_stages) e.g. (2, 8, 3, 3)

        if torch.isnan(attention_weights).any():
            attention_weights = torch.ones_like(attention_weights) / num_stages

        attention_weights = self.dropout(attention_weights)
        # attention_weights shape: (batch_size, num_heads, num_stages, num_stages) e.g. (2, 8, 3, 3)

        # Apply the attention weights
        attention_output = torch.matmul(attention_weights, V)
        # attention_output shape: (batch_size, num_heads, num_stages, head_dim) e.g. (2, 8, 3, 8)

        # Reshape back to the original form
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, num_stages, embed_dim)
        # transpose(1, 2) shape: (batch_size, num_stages, num_heads, head_dim) e.g. (2, 3, 8, 8)
        # view reshape shape: (batch_size, num_stages, stage_embed_dim) e.g. (2, 3, 64)

        # Output projection
        attention_output = self.W_o(attention_output)
        # attention_output shape: (batch_size, num_stages, stage_embed_dim) e.g. (2, 3, 64)

        # Residual connection + LayerNorm
        output = self.layer_norm(attention_output + stage_embeddings)
        # output shape: (batch_size, num_stages, stage_embed_dim) e.g. (2, 3, 64)

        if stage_valid_mask is not None:
            output = output * stage_valid_mask.unsqueeze(-1).to(output.dtype)

        return output


class RiceCPAT(nn.Module):
    """Stage-temporal attention encoder regressor.

    Built on the encoder-only architecture with stage-temporal processing:
    the input sequence is divided into several temporal stages, each stage is
    encoded independently, and cross-attention is then computed across the
    stages.

    Parameters
    ----------
    d_input:
        Model input dimension
    d_model:
        Input vector dimension
    d_output:
        Model output dimension (usually 1 for regression)
    q:
        Query and key dimension
    v:
        Value dimension
    h:
        Number of attention heads
    N:
        Number of encoder layers
    num_stages:
        Number of temporal stages
    stage_embed_dim:
        Stage embedding dimension
    attention_size:
        Attention window size; None means global attention. Default None
    dropout:
        Dropout probability after each MHA or PFF block. Default 0.3
    chunk_mode:
        Switching mode of the multi-head attention block. One of 'chunk',
        'window' or None. Default 'chunk'
    pe:
        Positional encoding type. One of 'original', 'regular' or None.
        Default None
    pe_period:
        Period for the 'regular' positional encoding. Default None
    pooling:
        Pooling method that aggregates the stage features to a single output.
        One of 'mean', 'max', 'last' or 'first'. Default 'mean'
    stage_pooling:
        Pooling method within each stage. One of 'mean', 'max', 'last' or
        'first'. Default 'mean'
    """

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int = 1,
                 q: int = 8,
                 v: int = 8,
                 h: int = 8,
                 N: int = 6,
                 num_stages: int = 3,
                 stage_embed_dim: int = None,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 pe_period: int = None,
                 pooling: str = 'mean',
                 stage_pooling: str = 'mean'):
        """Create the stage-temporal attention encoder."""
        super().__init__()

        self._d_model = d_model
        self._pooling = pooling
        self._num_stages = num_stages

        # If no stage embedding dimension is given, use d_model
        self._stage_embed_dim = stage_embed_dim if stage_embed_dim is not None else d_model

        # Input embedding layer
        self._embedding = nn.Linear(d_input, d_model)

        # Stage encoder
        self.stage_encoder = StageEncoder(d_model, self._stage_embed_dim, stage_pooling)

        # Stack of cross-stage attention layers
        self.staged_cross_attention_layers = nn.ModuleList([
            StagedCrossAttention(self._stage_embed_dim, h, dropout)
            for _ in range(N)
        ])

        # Stack of conventional encoder layers (optional, to enhance
        # intra-stage representations)
        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(max(1, N//2))])

        # Final output layer
        self._linear = nn.Linear(self._stage_embed_dim, d_output)

        # Positional encoding settings
        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        # Validate the pooling method
        valid_pooling = ['mean', 'max', 'last', 'first']
        if pooling not in valid_pooling:
            raise NameError(
                f'Pooling "{pooling}" not understood. Must be one of {", ".join(valid_pooling)}.')

        self.name = 'RiceCPAT'

    def _create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create a padding mask based on zero values in the input.

        Parameters
        ----------
        x:
            Input tensor (batch_size, seq_len, d_input)

        Returns
        -------
            Padding mask (batch_size, seq_len); 1 means real data, 0 means padding
        """
        # x shape: (batch_size, seq_len, d_input) e.g. (2, 12, 13)
        mask = torch.sum(torch.abs(x), dim=-1) != 0
        # torch.sum(torch.abs(x), dim=-1) shape: (batch_size, seq_len) e.g. (2, 12)
        # mask shape: (batch_size, seq_len) e.g. (2, 12) [bool]
        return mask.float()
        # return shape: (batch_size, seq_len) e.g. (2, 12) [float: 1.0 = real data, 0.0 = padding]

    def _split_sequence_into_stages(self, x: torch.Tensor, mask: torch.Tensor = None):
        """Split the sequence into several temporal stages.

        Parameters
        ----------
        x: torch.Tensor
            Input sequence (batch_size, seq_len, d_model)
        mask: torch.Tensor
            Padding mask (batch_size, seq_len)

        Returns
        -------
        stages: list
            List of the stage sequences
        stage_masks: list
            List of the stage masks
        """
        batch_size, seq_len, d_model = x.shape
        # x shape: (batch_size, seq_len, d_model) e.g. (2, 12, 64)

        # Compute the length of each stage
        stage_len = seq_len // self._num_stages  # e.g. 12 // 3 = 4
        remainder = seq_len % self._num_stages   # e.g. 12 % 3 = 0

        stages = []
        stage_masks = []

        start_idx = 0
        for i in range(self._num_stages):  # e.g. range(3) -> 0, 1, 2
            # If there is a remainder, the earlier stages get one extra step
            current_stage_len = stage_len + (1 if i < remainder else 0)  # e.g. 4
            end_idx = start_idx + current_stage_len  # e.g. 0+4=4, 4+4=8, 8+4=12

            stage = x[:, start_idx:end_idx, :]
            # stage shape: (batch_size, stage_len, d_model) e.g. (2, 4, 64)
            stages.append(stage)

            if mask is not None:
                stage_mask = mask[:, start_idx:end_idx]
                # stage_mask shape: (batch_size, stage_len) e.g. (2, 4)
                stage_masks.append(stage_mask)
            else:
                stage_masks.append(None)

            start_idx = end_idx

        # Returns:
        # stages: list of 3 tensors, each (batch_size, stage_len, d_model) e.g. 3 x (2, 4, 64)
        # stage_masks: list of 3 tensors, each (batch_size, stage_len) e.g. 3 x (2, 4)
        return stages, stage_masks

    def _split_sequence_by_stage_ids(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        stage_ids: torch.Tensor,
    ):
        """Split the sequence according to the per-sample phenological stage labels.

        stage_ids are aligned with the original time steps; 0/1/2 denote
        TP–PI, PI–FL and FL–MS respectively, and -1 denotes padding. Different
        samples may have different numbers of steps within the same stage.
        Before returning, each stage is padded independently within the stage
        and a new mask is generated.
        """
        if stage_ids.shape != mask.shape:
            raise ValueError(
                f"stage_ids shape {tuple(stage_ids.shape)} must match mask {tuple(mask.shape)}."
            )
        if stage_ids.device != x.device:
            stage_ids = stage_ids.to(x.device)

        real_steps = mask.bool()
        invalid_ids = real_steps & ((stage_ids < 0) | (stage_ids >= self._num_stages))
        if invalid_ids.any():
            bad_values = torch.unique(stage_ids[invalid_ids]).detach().cpu().tolist()
            raise ValueError(f"Real time steps contain invalid stage_ids: {bad_values}")

        batch_size, _, d_model = x.shape
        stages, stage_masks = [], []
        for stage_idx in range(self._num_stages):
            membership = real_steps & (stage_ids == stage_idx)
            lengths = membership.sum(dim=1)
            max_stage_len = max(1, int(lengths.max().item()))

            stage = x.new_zeros((batch_size, max_stage_len, d_model))
            stage_mask = mask.new_zeros((batch_size, max_stage_len))
            for batch_idx in range(batch_size):
                positions = torch.nonzero(membership[batch_idx], as_tuple=False).squeeze(-1)
                count = int(positions.numel())
                if count:
                    stage[batch_idx, :count] = x[batch_idx].index_select(0, positions)
                    stage_mask[batch_idx, :count] = 1.0

            stages.append(stage)
            stage_masks.append(stage_mask)

        return stages, stage_masks

    def _apply_final_pooling(
        self,
        stage_embeddings: torch.Tensor,
        stage_valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply the final pooling over the stage embeddings.

        Parameters
        ----------
        stage_embeddings: torch.Tensor
            Stage embeddings (batch_size, num_stages, stage_embed_dim)

        Returns
        -------
        pooled: torch.Tensor
            Pooled representation (batch_size, stage_embed_dim)
        """
        # stage_embeddings shape: (batch_size, num_stages, stage_embed_dim) e.g. (2, 3, 64)

        if stage_valid_mask is not None:
            valid = stage_valid_mask.bool()
            if self._pooling == 'mean':
                weights = valid.unsqueeze(-1).to(stage_embeddings.dtype)
                pooled = (stage_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
            elif self._pooling == 'max':
                masked = stage_embeddings.masked_fill(~valid.unsqueeze(-1), -torch.inf)
                pooled = masked.max(dim=1).values
                pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
            elif self._pooling == 'last':
                last_indices = (valid.long() * torch.arange(
                    1, valid.shape[1] + 1, device=valid.device
                )).argmax(dim=1)
                pooled = stage_embeddings[torch.arange(stage_embeddings.shape[0], device=valid.device), last_indices]
            elif self._pooling == 'first':
                first_indices = valid.to(torch.int64).argmax(dim=1)
                pooled = stage_embeddings[torch.arange(stage_embeddings.shape[0], device=valid.device), first_indices]
            else:
                weights = valid.unsqueeze(-1).to(stage_embeddings.dtype)
                pooled = (stage_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        elif self._pooling == 'mean':
            pooled = torch.mean(stage_embeddings, dim=1)
            # pooled shape: (batch_size, stage_embed_dim) e.g. (2, 64)
        elif self._pooling == 'max':
            pooled, _ = torch.max(stage_embeddings, dim=1)
            # pooled shape: (batch_size, stage_embed_dim) e.g. (2, 64)
        elif self._pooling == 'last':
            pooled = stage_embeddings[:, -1, :]
            # pooled shape: (batch_size, stage_embed_dim) e.g. (2, 64)
        elif self._pooling == 'first':
            pooled = stage_embeddings[:, 0, :]
            # pooled shape: (batch_size, stage_embed_dim) e.g. (2, 64)
        else:
            pooled = torch.mean(stage_embeddings, dim=1)
            # pooled shape: (batch_size, stage_embed_dim) e.g. (2, 64)

        return pooled

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_attention=False,
        stage_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Propagate the input through the stage-temporal attention encoder.

        Stage-temporal processing pipeline:
        1. Input embedding and positional encoding
        2. Conventional encoder layers to enhance the representation
        3. Split the sequence into several temporal stages
        4. Encode each stage independently into a fixed dimension
        5. Multi-layer cross-attention across the stages
        6. Final pooling and output

        Parameters
        ----------
        x:
            :class:`torch.Tensor` of shape (batch_size, K, d_input)
        mask:
            Optional padding mask of shape (batch_size, K).
            If None, a mask is created automatically from zero-valued inputs.
            1 means a real token, 0 means a padding token.
        return_attention:
            Whether to return the attention weights (for visualization)
        stage_ids:
            Optional phenological stage labels of shape (batch_size, seq_len).
            Real time steps use 0/1/2 for TP–PI, PI–FL, FL–MS; padding uses -1.
            When None, the original near-equal segmentation behavior is kept.

        Returns
        -------
            If return_attention=False: output tensor of shape (batch_size, d_output)
            If return_attention=True: (output, attention_weights)
        """
        batch_size, K, _ = x.shape
        # input shape: (batch_size, seq_len, d_input) e.g. (2, 12, 13)

        # Create the padding mask if none is provided
        if mask is None:
            mask = self._create_padding_mask(x)
            # mask shape: (batch_size, seq_len) e.g. (2, 12)

        # Embedding module - map the input features to the model hidden dimension
        encoding = self._embedding(x)
        # encoding shape: (batch_size, seq_len, d_model) e.g. (2, 12, 64)

        # Add the positional encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            # positional_encoding shape: (seq_len, d_model) e.g. (12, 64)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)
            # encoding shape: (batch_size, seq_len, d_model) e.g. (2, 12, 64) [positional info added]

        # Conventional encoder stack to enhance the representation (N//2 layers)
        for layer in self.layers_encoding:
            encoding = layer(encoding, mask=mask)
            # each layer input/output shape: (batch_size, seq_len, d_model) e.g. (2, 12, 64)

        # Split the sequence into temporal stages. When stage_ids is None the
        # original path is used, so the state_dict structure and the default
        # inference of legacy weights are unchanged.
        use_phenology_stages = stage_ids is not None
        if use_phenology_stages:
            stages, stage_masks = self._split_sequence_by_stage_ids(encoding, mask, stage_ids)
        else:
            stages, stage_masks = self._split_sequence_into_stages(encoding, mask)
        # stages: list of tensors, each (batch_size, stage_len, d_model) e.g. 3 x (2, 4, 64)
        # stage_masks: list of tensors, each (batch_size, stage_len) e.g. 3 x (2, 4)

        # Encode each stage into a fixed dimension
        stage_embeddings = []
        for stage, stage_mask in zip(stages, stage_masks):
            stage_embed = self.stage_encoder(
                stage,
                stage_mask,
                apply_attention_mask=use_phenology_stages,
            )
            # stage input shape: (batch_size, stage_len, d_model) e.g. (2, 4, 64)
            # stage_embed output shape: (batch_size, stage_embed_dim) e.g. (2, 64)
            stage_embeddings.append(stage_embed)

        # Stack into a stage-embedding tensor
        stage_embeddings = torch.stack(stage_embeddings, dim=1)
        # stage_embeddings shape: (batch_size, num_stages, stage_embed_dim) e.g. (2, 3, 64)

        stage_valid_mask = None
        if use_phenology_stages:
            stage_valid_mask = torch.stack(
                [stage_mask.bool().any(dim=1) for stage_mask in stage_masks],
                dim=1,
            )

        # Store the attention weights
        attention_weights = [] if return_attention else None

        # Cross-stage attention stack (N layers)
        for layer in self.staged_cross_attention_layers:
            stage_embeddings = layer(stage_embeddings, stage_valid_mask=stage_valid_mask)
            # each layer input/output shape: (batch_size, num_stages, stage_embed_dim) e.g. (2, 3, 64)

            # Collect the attention weights if requested
            if return_attention:
                # TODO: collect actual attention weights here
                attention_weights.append(None)  # placeholder

        # Final pooling - aggregate the stage embeddings into a single representation
        pooled = self._apply_final_pooling(stage_embeddings, stage_valid_mask=stage_valid_mask)
        # pooled shape: (batch_size, stage_embed_dim) e.g. (2, 64)

        # Final output layer - regression prediction
        output = self._linear(pooled)
        # output shape: (batch_size, d_output) e.g. (2, 1)

        if return_attention:
            return output, attention_weights
        else:
            return output


def analyze_model_architecture(model, model_name="RiceCPAT"):
    """Analyze the riceCPAT model architecture in detail."""
    print(f"\n{'='*80}")
    print(f"🔍 {model_name} architecture analysis")
    print(f"{'='*80}")

    # Basic information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"📊 Model overview:")
    print(f"   - model name: {model.name}")
    print(f"   - number of stages: {model._num_stages}")
    print(f"   - stage embedding dim: {model._stage_embed_dim}")
    print(f"   - total parameters: {total_params:,}")
    print(f"   - trainable parameters: {trainable_params:,}")
    print(f"   - model size: {total_params * 4 / (1024**2):.2f} MB (fp32)")

    # Parameter statistics per component
    print(f"\n📊 Parameters per component:")
    print("-" * 60)

    # Input embedding
    embedding_params = sum(p.numel() for p in model._embedding.parameters())
    print(f"Input embedding: {embedding_params:,} parameters")

    # Stage encoder
    stage_encoder_params = sum(p.numel() for p in model.stage_encoder.parameters())
    print(f"Stage encoder: {stage_encoder_params:,} parameters")

    # Conventional encoder layers
    traditional_encoder_params = sum(p.numel() for layer in model.layers_encoding for p in layer.parameters())
    print(f"Conventional encoder layers: {traditional_encoder_params:,} parameters")

    # Cross-attention layers
    cross_attention_params = sum(p.numel() for layer in model.staged_cross_attention_layers for p in layer.parameters())
    print(f"Cross-attention layers: {cross_attention_params:,} parameters")

    # Output layer
    output_params = sum(p.numel() for p in model._linear.parameters())
    print(f"Output layer: {output_params:,} parameters")

    print(f"{'='*80}\n")


def test_forward_pass(model, model_name="RiceCPAT"):
    """Test the forward pass of the stage-temporal model."""
    print(f"\n🚀 {model_name} forward pass test")
    print("-" * 60)

    # Create test data
    batch_size, seq_len, d_input = 2, 12, 13
    test_input = torch.randn(batch_size, seq_len, d_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Number of stages: {model._num_stages}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(test_input)

    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {model._linear.out_features})")

    # Test the stage splitting
    mask = model._create_padding_mask(test_input)
    embedded = model._embedding(test_input)
    stages, stage_masks = model._split_sequence_into_stages(embedded, mask)

    print(f"\n📊 Stage splitting result:")
    for i, (stage, stage_mask) in enumerate(zip(stages, stage_masks)):
        print(f"  Stage {i+1}: shape {stage.shape}, mask shape {stage_mask.shape if stage_mask is not None else 'None'}")

    print("-" * 60)


if __name__ == "__main__":
    print("🧠 RiceCPAT model architecture analysis")
    print("=" * 80)

    # Create the model
    model = RiceCPAT(
        d_input=13,
        d_model=64,
        d_output=1,
        q=8,
        v=8,
        h=8,
        N=4,
        num_stages=3,
        stage_embed_dim=64,
        attention_size=None,
        dropout=0.2,
        chunk_mode=None,
        pe="original",
        pooling='mean',
        stage_pooling='mean'
    )

    # Analyze the architecture
    analyze_model_architecture(model, "RiceCPAT")

    # Test the forward pass
    test_forward_pass(model, "RiceCPAT")

    print("✅ RiceCPAT analysis complete!")
