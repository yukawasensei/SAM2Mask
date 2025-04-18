# SAM2Mask

基于 Segment Anything Model 2 (SAM2) 的图像和视频分割工具，提供了便捷的 Web 界面，支持交互式图像和视频分割。

## 功能特点

- 🖼️ **图像分割**
  - 支持单张图片的前景/背景分割
  - 交互式标记点指定分割区域
  - 生成分割叠加效果和独立物品遮罩

- 🎥 **视频分割**
  - 支持视频文件的逐帧分割
  - 多物品标记和追踪
  - 自动生成分割效果视频

- 🚀 **技术优势**
  - 基于最新的 SAM2 模型
  - 支持 CUDA 加速（需要兼容的 GPU）
  - 用户友好的 Gradio Web 界面

## 环境要求

- Python 3.x
- CUDA 支持（推荐，也支持 CPU 模式）
- ffmpeg（用于视频处理）

## 安装步骤

1. 克隆仓库：
```bash
git clone [repository-url]
cd SAM2Mask
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 下载模型检查点：
- 将 SAM2 模型文件 `sam2_hiera_large.pt` 放置在 `checkpoints` 目录下
- 确保 `sam2_hiera_l.yaml` 配置文件存在

## 使用说明

1. **准备工作**
   - 创建 `images` 目录
   - 将待处理的图片或视频文件放入 `images` 目录

2. **启动应用**
```bash
python src/app.py
```

3. **图像分割**
   - 上传或选择示例图片
   - 使用标记工具标注前景/背景点
   - 点击"物品分割"生成结果

4. **视频分割**
   - 上传或选择视频文件
   - 为需要分割的物品添加标记点
   - 可以处理多个物品
   - 生成带有分割效果的输出视频

## 目录结构

```
SAM2Mask/
├── src/                # 源代码目录
│   ├── app.py         # 主程序入口
│   ├── image_segment.py    # 图像分割逻辑
│   ├── video_segment.py    # 视频分割逻辑
│   └── video_process.py    # 视频处理相关功能
├── checkpoints/       # 模型检查点目录
├── images/           # 输入图片和视频目录
└── output/           # 输出结果目录
```

## 注意事项

- 首次运行时需要下载模型文件
- 视频处理可能需要较长时间，请耐心等待
- 建议使用支持 CUDA 的 GPU 以获得更好的性能
- 当前版本为原型预览版本，UI 交互体验后续会持续优化

## 许可证

[添加许可证信息]

## 贡献指南

欢迎提交 Issue 和 Pull Request 来帮助改进项目。

## 致谢

- 感谢 SAM2 团队提供的优秀模型
- 感谢所有贡献者的支持 