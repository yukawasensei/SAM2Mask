import torch
from torch import nn

from src.sam2.modeling.sam2_utils import MLP

class SAM2Memory(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        memory_size: int = 10,
        memory_update_method: str = "gru",
        use_memory_mlp: bool = True,
        memory_mlp_layers: int = 2,
    ) -> None:
        """
        Memory module for SAM2.

        Args:
            embedding_dim: The channel dimension of the embeddings
            memory_size: The number of memory tokens
            memory_update_method: Method to update memory ("gru" or "attention")
            use_memory_mlp: Whether to use MLP to process memory tokens
            memory_mlp_layers: Number of layers in memory MLP
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.memory_update_method = memory_update_method
        self.use_memory_mlp = use_memory_mlp

        # Initialize memory tokens
        self.memory_tokens = nn.Parameter(torch.randn(1, memory_size, embedding_dim))

        if memory_update_method == "gru":
            # GRU for memory update
            self.memory_gru = nn.GRUCell(embedding_dim, embedding_dim)
        elif memory_update_method == "attention":
            # Attention for memory update
            self.memory_q = nn.Linear(embedding_dim, embedding_dim)
            self.memory_k = nn.Linear(embedding_dim, embedding_dim)
            self.memory_v = nn.Linear(embedding_dim, embedding_dim)
            self.memory_out = nn.Linear(embedding_dim, embedding_dim)
        else:
            raise ValueError(f"Unknown memory update method: {memory_update_method}")

        if use_memory_mlp:
            # MLP to process memory tokens
            self.memory_mlp = MLP(
                input_dim=embedding_dim,
                hidden_dim=embedding_dim,
                output_dim=embedding_dim,
                num_layers=memory_mlp_layers,
            )

    def forward(
        self,
        input_tokens: torch.Tensor,
        memory_tokens: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update memory tokens with input tokens.

        Args:
            input_tokens: Input tokens of shape (B, N, C)
            memory_tokens: Optional memory tokens of shape (B, M, C)

        Returns:
            Tuple containing:
            - Updated memory tokens of shape (B, M, C)
            - Memory attention output of shape (B, N, C)
        """
        B = input_tokens.shape[0]

        # Initialize or expand memory tokens
        if memory_tokens is None:
            memory_tokens = self.memory_tokens.expand(B, -1, -1)

        if self.memory_update_method == "gru":
            # Update memory using GRU
            memory_flat = memory_tokens.reshape(-1, self.embedding_dim)
            input_flat = input_tokens.reshape(-1, self.embedding_dim)
            memory_flat = self.memory_gru(input_flat, memory_flat)
            memory_tokens = memory_flat.reshape(B, -1, self.embedding_dim)

        elif self.memory_update_method == "attention":
            # Update memory using attention
            q = self.memory_q(memory_tokens)
            k = self.memory_k(input_tokens)
            v = self.memory_v(input_tokens)

            attn = torch.matmul(q, k.transpose(-2, -1)) / self.embedding_dim ** 0.5
            attn = torch.softmax(attn, dim=-1)

            memory_update = torch.matmul(attn, v)
            memory_tokens = memory_tokens + self.memory_out(memory_update)

        if self.use_memory_mlp:
            # Process memory tokens with MLP
            memory_tokens = self.memory_mlp(memory_tokens)

        # Compute attention between input and memory
        q = input_tokens @ memory_tokens.transpose(-2, -1) / self.embedding_dim ** 0.5
        attn = torch.softmax(q, dim=-1)
        memory_out = torch.matmul(attn, memory_tokens)

        return memory_tokens, memory_out 