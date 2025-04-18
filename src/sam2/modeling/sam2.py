import torch
from torch import nn

from src.sam2.modeling.sam2_base import SAM2Base
from src.sam2.modeling.sam2_utils import get_obj_ptrs_from_masks, get_tpos_enc

class SAM2(SAM2Base):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def device(self) -> torch.device:
        return self.image_encoder.patch_embed.proj.weight.device

    def forward(
        self,
        batched_input: list[dict],
        multimask_output: bool = False,
    ) -> list[dict]:
        """
        Predicts masks for the given input prompts and images.

        Args:
            batched_input: A list over input images, each a dictionary with the following keys:
                'image': The image as a torch tensor in 3xHxW format
                'original_size': (h, w) original size of the image
                'point_coords': Optional point coordinates in [N, 2] format
                'point_labels': Optional point labels in [N] format
                'boxes': Optional box coordinates in [N, 4] format
                'mask_inputs': Optional mask inputs in [N, H, W] format
                'obj_ptrs': Optional object pointer coordinates in [T, 2] format
                'tpos_enc': Optional temporal positional encoding in [T, 1] format
            multimask_output: Whether to return multiple masks per input

        Returns:
            A list over input images, each a dictionary with the following keys:
                'masks': Predicted masks in [num_masks, H, W] format
                'iou_predictions': Predicted IoU scores in [num_masks] format
                'low_res_masks': Low resolution masks in [num_masks, H, W] format
                'obj_scores': Optional object scores in [1] format
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings, high_res_features = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"],)
                labels = (image_record["point_labels"],)
            else:
                points = None
                labels = None

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
                obj_ptrs=image_record.get("obj_ptrs", None),
                tpos_enc=image_record.get("tpos_enc", None),
            )

            low_res_masks, iou_predictions, mask_tokens, obj_scores = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                high_res_features=high_res_features,
            )

            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )

            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_masks": low_res_masks,
                    "obj_scores": obj_scores,
                }
            )

        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: tuple[int, ...],
        original_size: tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks: Batched masks from the mask decoder
            input_size: Input size to remove padding
            original_size: Original image size to upscale to

        Returns:
            Unpadded and upscaled masks
        """
        masks = nn.functional.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = nn.functional.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh, padw = self.image_encoder.img_size - h, self.image_encoder.img_size - w
        x = nn.functional.pad(x, (0, padw, 0, padh))
        return x 