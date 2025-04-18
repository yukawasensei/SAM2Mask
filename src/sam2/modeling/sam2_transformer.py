import torch
from torch import Tensor, nn

from src.sam2.modeling.sam.transformer import TwoWayTransformer

class SAM2TwoWayTransformer(TwoWayTransformer):
    def __init__(
        self,
        depth: int = 12,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        activation: nn.Module = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        use_memory_attention: bool = True,
        memory_attention_position: str = "after_self",
        memory_attention_layers: list[int] = None,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
            depth: Number of layers in the transformer
            embedding_dim: The channel dimension for the input embeddings
            num_heads: The number of heads for multihead attention
            mlp_dim: The hidden dimension of the MLP block
            activation: The activation function to use
            attention_downsample_rate: The downsample rate for the attention blocks
            skip_first_layer_pe: Whether to skip the PE in the first layer
            use_memory_attention: Whether to use memory attention
            memory_attention_position: Where to add memory attention ("before_self", "after_self", or "parallel")
            memory_attention_layers: List of layer indices to add memory attention to
        """
        super().__init__(
            depth=depth,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            activation=activation,
            attention_downsample_rate=attention_downsample_rate,
            skip_first_layer_pe=skip_first_layer_pe,
        )

        self.use_memory_attention = use_memory_attention
        self.memory_attention_position = memory_attention_position
        self.memory_attention_layers = memory_attention_layers or list(range(depth))

        if use_memory_attention:
            # Add memory attention layers
            self.memory_attention = nn.ModuleList([
                MemoryAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                )
                for _ in range(depth)
            ])

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
        memory_tokens: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            image_embedding: Image to attend to
            image_pe: PE applied to image
            point_embedding: Query points
            memory_tokens: Memory tokens to attend to

        Returns:
            Tuple containing:
            - The processed point_embedding
            - The processed image_embedding
        """
        # BxNxC and BxHWxC -> BxNxC and BxHWxC
        point_embedding = point_embedding + self.point_embedding
        bs = image_embedding.shape[0]

        # Run the transformer
        for i in range(self.depth):
            if i == 0 and self.skip_first_layer_pe:
                src = image_embedding
                pos_src = None
                pos_point = None
            else:
                src = image_embedding
                pos_src = image_pe
                pos_point = point_embedding

            # Apply self attention to both point and image embeddings
            point_embedding = self.self_attn_layers[i](point_embedding, pos_point)
            image_embedding = self.self_attn_layers[i](src, pos_src)

            # Apply memory attention if enabled
            if (
                self.use_memory_attention
                and memory_tokens is not None
                and i in self.memory_attention_layers
            ):
                if self.memory_attention_position == "before_self":
                    point_embedding = self.memory_attention[i](point_embedding, memory_tokens)
                    image_embedding = self.memory_attention[i](image_embedding, memory_tokens)

                elif self.memory_attention_position == "after_self":
                    point_embedding = self.memory_attention[i](point_embedding, memory_tokens)
                    image_embedding = self.memory_attention[i](image_embedding, memory_tokens)

                elif self.memory_attention_position == "parallel":
                    # Apply memory attention in parallel with self attention
                    point_mem = self.memory_attention[i](point_embedding, memory_tokens)
                    image_mem = self.memory_attention[i](image_embedding, memory_tokens)
                    point_embedding = point_embedding + point_mem
                    image_embedding = image_embedding + image_mem

            # Apply cross attention
            point_embedding = self.cross_attn_layers[i](point_embedding, src, pos_src, pos_point)

        return point_embedding, image_embedding


class MemoryAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: nn.Module = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer block that attends to memory tokens.

        Args:
            embedding_dim: The channel dimension of the embeddings
            num_heads: The number of heads in the attention layers
            mlp_dim: The hidden dimension of the MLP block
            activation: The activation to use in the MLP block
            attention_downsample_rate: The downsample rate to use for attention
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.attention_downsample_rate = attention_downsample_rate

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            activation(),
            nn.Linear(mlp_dim, embedding_dim),
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
    ) -> Tensor:
        """
        Apply memory attention to the input queries.

        Args:
            queries: Input queries of shape (B, N, C)
            memory: Memory tokens of shape (B, M, C)

        Returns:
            Updated queries of shape (B, N, C)
        """
        # Apply attention
        q = self.q_proj(queries)
        k = self.k_proj(memory)
        v = self.v_proj(memory)

        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.embedding_dim // self.num_heads) ** 0.5
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(out.shape[0], -1, self.embedding_dim)
        out = self.out_proj(out)

        # First residual connection
        queries = queries + out

        # MLP
        queries = queries + self.mlp(self.norm2(queries))

        return queries 