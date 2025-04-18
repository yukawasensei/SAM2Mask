import torch
from torch import nn

from src.sam2.modeling.sam.prompt_encoder import PromptEncoder
from src.sam2.modeling.sam2_utils import MLP

class SAM2PromptEncoder(PromptEncoder):
    def __init__(
        self,
        *args,
        use_obj_ptrs=True,
        max_num_obj_ptrs=10,
        proj_tpos_enc_in_obj_ptrs=True,
        use_signed_tpos_enc_to_obj_ptrs=True,
        only_obj_ptrs_in_the_past_for_eval=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_obj_ptrs = use_obj_ptrs
        self.max_num_obj_ptrs = max_num_obj_ptrs
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        if use_obj_ptrs:
            self.obj_ptr_embed = MLP(
                input_dim=2,  # x, y coordinates
                hidden_dim=self.embed_dim,
                output_dim=self.embed_dim,
                num_layers=2,
            )
            if proj_tpos_enc_in_obj_ptrs:
                self.tpos_enc_proj = nn.Linear(1, self.embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the size of the image.
        Returns (1, embed_dim, H, W) or (1, H, W, embed_dim).
        """
        return self.pe_layer.forward()

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _embed_obj_ptrs(
        self,
        obj_ptrs: torch.Tensor,
        tpos_enc: torch.Tensor = None,
        is_eval: bool = False,
    ) -> torch.Tensor:
        """
        Embeds object pointers.

        Args:
            obj_ptrs: Object pointer coordinates of shape (B, T, 2)
            tpos_enc: Temporal positional encoding of shape (B, T, 1)
            is_eval: Whether in evaluation mode

        Returns:
            Object pointer embeddings of shape (B, T, C)
        """
        if not self.use_obj_ptrs:
            return None

        # Get base embeddings from coordinates
        obj_ptr_embedding = self.obj_ptr_embed(obj_ptrs)

        if self.proj_tpos_enc_in_obj_ptrs and tpos_enc is not None:
            # Project and add temporal positional encoding
            if self.use_signed_tpos_enc_to_obj_ptrs:
                # Use signed temporal encoding
                tpos_enc_proj = self.tpos_enc_proj(tpos_enc)
            else:
                # Use absolute temporal encoding
                tpos_enc_proj = self.tpos_enc_proj(torch.abs(tpos_enc))
            obj_ptr_embedding = obj_ptr_embedding + tpos_enc_proj

        if is_eval and self.only_obj_ptrs_in_the_past_for_eval:
            # Zero out future object pointers during evaluation
            future_mask = tpos_enc > 0
            obj_ptr_embedding[future_mask.squeeze(-1)] = 0

        return obj_ptr_embedding

    def forward(
        self,
        points: torch.Tensor = None,
        boxes: torch.Tensor = None,
        masks: torch.Tensor = None,
        obj_ptrs: torch.Tensor = None,
        tpos_enc: torch.Tensor = None,
        is_eval: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
            points: Point coordinates and labels of shape (B, N, 2) and (B, N)
            boxes: Box coordinates of shape (B, N, 2, 2)
            masks: Mask inputs of shape (B, 1, H, W)
            obj_ptrs: Object pointer coordinates of shape (B, T, 2)
            tpos_enc: Temporal positional encoding of shape (B, T, 1)
            is_eval: Whether in evaluation mode

        Returns:
            Sparse embeddings of shape (B, N, C) and dense embeddings of shape (B, C, H, W)
        """
        sparse_embeddings = torch.empty((0, 0, self.embed_dim), device=points.device)
        if points is not None:
            points_embeddings = self._embed_points(points, labels, pad=True)
            sparse_embeddings = torch.cat([sparse_embeddings, points_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if obj_ptrs is not None:
            obj_ptr_embeddings = self._embed_obj_ptrs(obj_ptrs, tpos_enc, is_eval)
            if obj_ptr_embeddings is not None:
                sparse_embeddings = torch.cat([sparse_embeddings, obj_ptr_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                points.shape[0], -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings 