import torch
from torch import nn

from src.sam2.modeling.sam.image_encoder import ImageEncoderViT
from src.sam2.modeling.sam2_utils import MLP

class SAM2ImageEncoder(ImageEncoderViT):
    def __init__(
        self,
        *args,
        use_high_res_features=True,
        use_obj_ptrs_in_encoder=True,
        max_num_obj_ptrs=10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_high_res_features = use_high_res_features
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_num_obj_ptrs = max_num_obj_ptrs

        if use_obj_ptrs_in_encoder:
            self.obj_ptr_embed = MLP(
                input_dim=2,  # x, y coordinates
                hidden_dim=self.embed_dim,
                output_dim=self.embed_dim,
                num_layers=2,
            )

    def forward(
        self,
        x: torch.Tensor,
        obj_ptrs: torch.Tensor = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass of the image encoder.

        Args:
            x: Input image tensor of shape (B, 3, H, W)
            obj_ptrs: Object pointer coordinates of shape (B, T, 2)

        Returns:
            Tuple containing:
            - Image embeddings of shape (B, C, H//16, W//16)
            - List of high-resolution feature maps if use_high_res_features is True
        """
        high_res_features = []

        # Patch embedding
        x = self.patch_embed(x)

        # Add positional embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # Add object pointer embeddings if enabled
        if self.use_obj_ptrs_in_encoder and obj_ptrs is not None:
            # Embed object pointers
            obj_ptr_embedding = self.obj_ptr_embed(obj_ptrs)
            B, T, C = obj_ptr_embedding.shape

            # Reshape object pointer embeddings to match image embedding shape
            obj_ptr_embedding = obj_ptr_embedding.view(B, T, 1, 1).expand(
                -1, -1, x.shape[2], x.shape[3]
            )

            # Add object pointer embeddings to image embeddings
            x = x + obj_ptr_embedding

        # Apply transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.use_high_res_features and i in [3, 7, 11]:  # Collect features from different layers
                high_res_features.append(x.clone())

        # Apply final normalization
        x = self.neck(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_high_res_features:
            return x, high_res_features
        else:
            return x, None 