import torch
from torch import nn
from torch.nn import functional as F

from src.sam2.modeling.sam.mask_decoder import MaskDecoder
from src.sam2.modeling.sam2_utils import MLP

class SAM2MaskDecoder(MaskDecoder):
    def __init__(
        self,
        *args,
        use_high_res_features=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_high_res_features = use_high_res_features
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        if pred_obj_scores_mlp:
            self.obj_score_mlp = MLP(
                input_dim=self.transformer_dim,
                hidden_dim=self.transformer_dim,
                output_dim=1,
                num_layers=3,
            )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool = True,
        high_res_features: list = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): A tensor of shape
            (B, num_image_tokens, C) or (B, C, H, W) containing
            the image embeddings
          image_pe (torch.Tensor): A tensor of shape (1, num_image_tokens, C)
            containing the positional encoding
          sparse_prompt_embeddings (torch.Tensor): A tensor of shape
            (B, num_sparse_prompts, C) containing the sparse prompt
            embeddings
          dense_prompt_embeddings (torch.Tensor): A tensor of shape
            (B, num_dense_prompts, C) containing the dense prompt
            embeddings
          multimask_output (bool): Whether to return multiple masks or
            a single mask
          repeat_image (bool): Whether to repeat the image embeddings
            for each prompt (default: True)
          high_res_features (list): A list of high-resolution feature maps
            from the image encoder (default: None)

        Returns:
          torch.Tensor: A tensor of shape (B, num_masks, H, W)
            containing the predicted masks
          torch.Tensor: A tensor of shape (B, num_masks) containing
            the predicted iou scores
          torch.Tensor: A tensor of shape (B, num_masks, C)
            containing the mask tokens
        """
        # Concatenate output tokens
        output_tokens = torch.cat([sparse_prompt_embeddings, dense_prompt_embeddings], dim=1)

        if len(image_embeddings.shape) == 4:
            # If image_embeddings has shape (B, C, H, W), flatten it
            B, C, H, W = image_embeddings.shape
            image_embeddings = image_embeddings.flatten(2).permute(0, 2, 1)
            image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, sparse_prompt_embeddings.shape[0], dim=0)
            pos_src = torch.repeat_interleave(image_pe, sparse_prompt_embeddings.shape[0], dim=0)
        else:
            src = image_embeddings
            pos_src = image_pe

        b, c, h, w = src.shape[0], self.transformer_dim, int(src.shape[1] ** 0.5), int(src.shape[1] ** 0.5)

        # Run the transformer
        hs, src = self.transformer(src, pos_src, output_tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.permute(0, 2, 1).view(b, c, h, w)

        if self.use_high_res_features and high_res_features is not None:
            # Use high-resolution feature maps from the image encoder
            upscaled_embedding_0 = high_res_features[0]  # level 0 features
            upscaled_embedding_1 = high_res_features[1]  # level 1 features
            upscaled_embedding_2 = src  # level 2 features
        else:
            # Use only the lowest resolution feature map
            upscaled_embedding_0 = None
            upscaled_embedding_1 = None
            upscaled_embedding_2 = src

        hyper_in_list: list[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_2.shape

        # Generate masks
        masks = self.output_upscaling(
            upscaled_embedding_0,
            upscaled_embedding_1,
            upscaled_embedding_2,
            hyper_in,
        )

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the most appropriate mask or masks for output
        if multimask_output:
            # Return all masks if multimask_output is True
            mask_slice = slice(1, None)
        else:
            # Return the first mask if multimask_output is False
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Predict object scores if enabled
        if self.pred_obj_scores:
            if self.pred_obj_scores_mlp:
                # Use an MLP to predict object scores
                object_score_logits = self.obj_score_mlp(iou_token_out)
            else:
                # Use the first IoU prediction as the object score
                object_score_logits = iou_pred[:, 0:1]
        else:
            # Return None if object score prediction is disabled
            object_score_logits = None

        return masks, iou_pred, mask_tokens_out, object_score_logits 