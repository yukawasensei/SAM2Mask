import os
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from video_process import PointSet
from glob import glob
from PIL import Image, ImageDraw
import gc
import cv2
# from segment_anything import sam_model_registry, SamPredictor # 不再需要

# from sam2.build_sam import build_sam2_video_predictor # 不再直接调用
from sam2.build_sam import _load_checkpoint # 需要加载 checkpoint 的函数
from hydra import compose
from hydra.initialize import initialize_config_dir
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra # <-- 导入 GlobalHydra

# 获取当前文件所在目录
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_CURRENT_DIR) # 项目根目录是上一级

# 检测设备类型
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 构建绝对路径
sam2_checkpoint = os.path.join(_ROOT_DIR, "checkpoints/sam2.1_hiera_large.pt")
# model_cfg 现在指向包含配置文件的目录的父目录 (Hydra 需要 config_dir 指向 configs 所在的目录)
config_path = os.path.join(_CURRENT_DIR, 'sam2', 'configs')
# 配置名保持不变，Hydra 会在 config_path 下搜索
model_cfg_name = "sam2.1/sam2.1_hiera_l" # 这是 config_name

# ---- 检查 Hydra 是否已初始化，如果需要则清理 ----
if GlobalHydra().is_initialized():
    GlobalHydra.instance().clear()
# ---- End ----

# 初始化 Hydra，指定配置目录
with initialize_config_dir(config_dir=os.path.abspath(config_path), version_base=None):
    # 加载配置
    cfg = compose(config_name=model_cfg_name)
    OmegaConf.resolve(cfg)
    # 覆盖目标类及其他参数 (与 build_sam.py 中的保持一致)
    hydra_overrides = [
        "model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
        # 其他在 build_sam.py 中添加的 overrides
        "model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
        "model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
        "model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        "model.binarize_mask_from_pts_for_mem_enc=true",
        "model.fill_hole_area=8",
    ]
    # 应用覆盖 - 使用 OmegaConf.update 逐条更新
    # cfg_override = OmegaConf.from_dotlist(hydra_overrides)
    # cfg = OmegaConf.merge(cfg, cfg_override) # 不再使用 merge
    OmegaConf.set_struct(cfg, False)
    for override in hydra_overrides:
        key, value = override.split('=', 1)
        # 尝试解析值为 Python 类型 (bool, int, float)
        if value.lower() == 'true':
            parsed_value = True
        elif value.lower() == 'false':
            parsed_value = False
        else:
            try:
                parsed_value = int(value)
            except ValueError:
                try:
                    parsed_value = float(value)
                except ValueError:
                    parsed_value = value # 保持为字符串
        OmegaConf.update(cfg, key, parsed_value, merge=True) # 使用 update

    # 实例化模型
    video_predicator = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(video_predicator, sam2_checkpoint)
    video_predicator = video_predicator.to(device)
    video_predicator.eval()


class InterferenceFrame:

    def __init__(self):        
        self.origin_frame_id = 0
        self.item_id = 0
        self.point_set = None


def prepare_path(base_path):
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M%S")
    results = []
    results.append(os.path.join(base_path, date_str, time_str, 'origin'))
    results.append(os.path.join(base_path, date_str, time_str, 'mask'))
    results.append(os.path.join(base_path, date_str, time_str, 'result'))
    for d in results:
        os.makedirs(d, exist_ok = True)
    results.append(os.path.join(base_path, date_str, f'{time_str}.mp4'))
    return results


def preprare_video_frames(source_video, frame_path):
    command = f'ffmpeg -i {source_video} -q:v 2 -start_number 0 {os.path.join(frame_path, "%05d.jpg")} 2>&1'
    with os.popen(command) as fp:
        fp.readlines()


def merge_mask(origin_file, mask_file, result_file):
    origin_file = os.path.abspath(origin_file)
    mask_file = os.path.abspath(mask_file)
    result_file = os.path.abspath(result_file)
    command = f'ffmpeg -i {mask_file} -i {origin_file} -filter_complex "[0][1]blend=all_expr=0.3*A+0.7*B" {result_file} 2>&1 '
    print(f'command = {command}')
    with os.popen(command) as fp:
        fp.readlines()

def merge_video(result_dir, result_file):
    command = f'ffmpeg -f image2 -i {os.path.join(result_dir, "%05d.jpg")} {result_file}'
    with os.popen(command) as fp:
        fp.readlines()


def video_interfrence(video_path, output_path, frames, width, height):
    # prepare pathes
    origin_path, mask_path, result_path, final_file = prepare_path(output_path)
    # extract frames from video
    preprare_video_frames(video_path, origin_path)
    # initialize predicator
    inference_state = video_predicator.init_state(video_path=origin_path)
    for f in frames:
        params = {
            'inference_state': inference_state,
            'frame_idx': f.origin_frame_id,
            'obj_id': f.item_id,
            'points': [(x[0], x[1]) for x in f.point_set],
            'labels': [ x[2] for x in f.point_set]
        }
        video_predicator.add_new_points_or_box(**params)

    video_segments = {}  # video_segments contains the per-frame segmentation results
    # propagation
    for out_frame_idx, out_obj_ids, out_mask_logits in tqdm(video_predicator.propagate_in_video(inference_state), desc='video propagation'):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    # combine the result and visualization
    color_mask_dict = {}
    for out_frame_idx in tqdm(range(len(video_segments)), desc='merge masks'):
        mask_all = np.ones((width, height, 3))
        if out_frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                if out_obj_id not in color_mask_dict:
                    color_mask_dict[out_obj_id] = np.random.random((1, 3)).tolist()[0]
                color_mask = color_mask_dict[out_obj_id]
                for i in range(3):
                    mask_all[out_mask[0] == True, i] = color_mask[i]
        img = Image.fromarray(np.uint8(mask_all * 255)).convert('RGB')
        file_name = '%05d.jpg' % out_frame_idx
        full_file_name = os.path.join(mask_path, file_name)
        full_origin_name = os.path.join(origin_path, file_name)
        full_result_name = os.path.join(result_path, file_name)
        img.save(full_file_name, format='JPEG')
        if os.path.exists(full_origin_name):
            merge_mask(full_origin_name, full_file_name, full_result_name)
    merge_video(result_path, final_file)
    # 清理内存
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return final_file
        
            
                
            
            
class VideoSegmentor:
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
    
    def process_frame(self, frame):
        """处理单帧图像
        
        Args:
            frame: numpy array 格式的图像帧
            
        Returns:
            numpy array: 处理后的图像帧
        """
        # 确保图片是 RGB 格式
        if len(frame.shape) == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[2] == 4:
            frame = frame[..., :3]
            
        # 设置图片
        self.predictor.set_image(frame)
        
        # 生成提示点（这里使用图片中心点作为示例）
        h, w = frame.shape[:2]
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
        result = frame.copy()
        result[mask] = result[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
        
        return result
    
    def process_video(self, video_path):
        """处理视频文件
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            str: 处理后的视频文件路径
        """
        # 读取视频
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 准备输出视频
        output_path = os.path.join("output", "processed_" + os.path.basename(video_path))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 处理每一帧
        with tqdm(total=total_frames, desc="处理视频") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 处理帧
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = self.process_frame(frame_rgb)
                processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                
                # 写入帧
                out.write(processed_frame_bgr)
                pbar.update(1)
        
        # 释放资源
        cap.release()
        out.release()
        
        return output_path
        
            
                
            
            