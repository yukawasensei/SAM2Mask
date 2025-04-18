import os
import torch
import numpy as np
import cv2
from PIL import Image
from src.sam2.build_sam import build_sam2
from src.sam2.sam2_image_predictor import SAM2ImagePredictor
from src.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

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
