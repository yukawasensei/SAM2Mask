import os
import torch
import numpy as np
import cv2
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from segment_anything import sam_model_registry, SamPredictor

# 全局变量
sam2_model = None
image_predictor = None

# 检测设备类型
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

def get_sam2_model(device):
    global sam2_model
    if sam2_model:
        return sam2_model
    model_cfg = "sam2_hiera_l.yaml"
    sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    return sam2_model

def get_image_predicator(device):
    global image_predictor
    if image_predictor:
        return image_predictor
    image_predictor = SAM2ImagePredictor(get_sam2_model(device))
    return image_predictor

def segment_one(input_image, predictor):
    """处理单张图片的分割"""
    # 读取图片
    if isinstance(input_image, str):
        image = Image.open(input_image).convert('RGB')
        image = np.array(image)
    else:
        image = input_image
        
    # 设置图片
    predictor.set_image(image)
    
    # 生成点击位置（图片中心点）
    h, w = image.shape[:2]
    point_coords = np.array([[w//2, h//2]])
    point_labels = np.array([1])
    
    # 预测掩码
    masks, iou_predictions, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    
    # 选择最佳掩码
    mask_idx = np.argmax(iou_predictions)
    mask = masks[mask_idx]
    
    # 创建彩色掩码
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[mask] = [255, 0, 0]  # 红色掩码
    
    # 合并原图和掩码
    result = image.copy()
    mask_region = mask[..., None]
    result = np.where(mask_region, cv2.addWeighted(result, 0.5, color_mask, 0.5, 0), result)
    
    return result, mask

def process_image(image_path, output_path):
    """处理图片并保存结果"""
    predictor = get_image_predicator(device)
    result, mask = segment_one(image_path, predictor)
    
    # 保存结果
    result_img = Image.fromarray(result)
    result_img.save(output_path)
    
    return output_path

def generator_inference(device, input_image):
    """使用自动掩码生成器处理图片"""
    mask_generator = SAM2AutomaticMaskGenerator(get_sam2_model(device))
    result, mask_all = segment_one(input_image, mask_generator)
    return result, mask_all

def image_inference(device, image, points=None):
    """处理图片推理
    
    Args:
        device: 设备类型
        image: 输入图片
        points: 点击位置和标签列表，格式为[((x, y), label), ...]
        
    Returns:
        tuple: (叠加结果图片, 遮罩图片)
    """
    predictor = get_image_predicator(device)
    
    # 设置图片
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
        image = np.array(image)
    predictor.set_image(image)
    
    # 处理点击位置
    if points and len(points) > 0:
        point_coords = np.array([p[0] for p in points])
        point_labels = np.array([p[1] for p in points])
    else:
        h, w = image.shape[:2]
        point_coords = np.array([[w//2, h//2]])
        point_labels = np.array([1])
    
    # 预测掩码
    masks, iou_predictions, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    
    # 选择最佳掩码
    mask_idx = np.argmax(iou_predictions)
    mask = masks[mask_idx]
    
    # 创建彩色掩码
    h, w = image.shape[:2]
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[mask] = [255, 0, 0]  # 红色掩码
    
    # 生成叠加结果
    result = image.copy()
    mask_region = mask[..., None]
    result = np.where(mask_region, cv2.addWeighted(result, 0.5, color_mask, 0.5, 0), result)
    
    # 生成遮罩图片
    mask_image = np.zeros_like(image)
    mask_image[mask] = [255, 255, 255]
    
    return result, mask_image

class ImageSegmentor:
    def __init__(self, model_type="vit_h", checkpoint="checkpoints/sam2_hiera_large.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.predictor = None
        self.initialize_model()
    
    def initialize_model(self):
        """初始化 SAM 模型"""
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(f"模型文件 {self.checkpoint} 不存在")
        
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
    
    def process_image(self, image):
        """处理单张图片
        
        Args:
            image: 输入图片（PIL Image 或 numpy array）
            
        Returns:
            tuple: (叠加结果图片, 遮罩图片)
        """
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # 确保图片是 RGB 格式
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[..., :3]
            
        # 设置图片
        self.predictor.set_image(image)
        
        # 生成提示点（这里使用图片中心点作为示例）
        h, w = image.shape[:2]
        input_point = np.array([[w//2, h//2]])
        input_label = np.array([1])
        
        # 生成遮罩
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        # 选择最佳遮罩
        mask = masks[np.argmax(scores)]
        
        # 生成叠加结果
        overlay = image.copy()
        overlay[mask] = overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
        
        # 生成遮罩图片
        mask_image = np.zeros_like(image)
        mask_image[mask] = [255, 255, 255]
        
        return overlay, mask_image
