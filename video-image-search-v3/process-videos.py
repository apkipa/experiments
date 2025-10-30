"""
输入视频路径，通过 ffmpeg 提取视频帧，然后使用 DINOv2 自动提取特征到 `workdir`。
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
from PIL import Image
import numpy as np
import onnxruntime as ort
from config import SCALE_FACTOR, USE_GPU, ADVANCED_SPLIT


def extract_frames(video_path: str, fps: int = 1) -> List[np.ndarray]:
    """使用ffmpeg pipe直接读取视频帧到numpy数组，不经过磁盘。"""

    # 获取视频宽高
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        video_path,
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe.stderr}")
    wh = probe.stdout.strip().split(",")
    width, height = int(wh[0]), int(wh[1])

    # 用ffmpeg pipe输出raw RGB帧
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
        "-f",
        "image2pipe",
        "-pix_fmt",
        "rgb24",
        "-vcodec",
        "rawvideo",
        "-",
    ]
    # proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)
    frame_size = width * height * 3
    frames = []
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
        frames.append(frame)
    proc.stdout.close()
    proc.wait()
    print(
        f"Extracted {len(frames)} frames from `{video_path}` to memory via ffmpeg pipe."
    )
    return frames


def load_image_from_array(image_array: np.ndarray) -> Image.Image:
    """从numpy数组加载PIL图片。"""
    return Image.fromarray(image_array)


def preprocess_image(image: Image.Image, target_size: int = 224) -> np.ndarray:
    """预处理图片为DINOv2模型输入。"""
    # Resize + BICUBIC
    image = image.resize((target_size, target_size), Image.Resampling.BICUBIC)
    # 转为RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image).astype("float32") / 255.0
    # ImageNet归一化参数 for DINOv2
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # HWC to CHW
    arr = np.expand_dims(arr, axis=0)  # Add batch dimension
    arr = arr.astype("float32")  # 保证为float32
    return arr


def extract_features_from_arrays(
    image_arrays: List[np.ndarray], model_path: Optional[str] = None
) -> np.ndarray:
    """直接从帧数组提取特征，每个帧生成变体（块或裁剪）。"""
    from tqdm import tqdm
    
    # 设置 ONNX providers
    if USE_GPU:
        providers = ['CUDAExecutionProvider', 'OpenVINOExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    if model_path is None:
        model_path = "models/dinov2-small/model.onnx"
    
    session = ort.InferenceSession(model_path, providers=providers)
    features = []
    for image_array in tqdm(image_arrays, desc="提取特征", ncols=80):
        image = load_image_from_array(image_array)
        variants = generate_image_variants(image)
        frame_features = []
        for variant in variants:
            input_tensor = preprocess_image(variant)
            outputs = session.run(None, {"pixel_values": input_tensor})
            # DINOv2输出last_hidden_state，平均池化
            # print(outputs[0].shape)
            # feature = outputs[0].mean(axis=1).squeeze()
            feature = outputs[0][:, 0, :].squeeze()  # 形状: (384,)
            frame_features.append(feature)
        features.append(frame_features)
    return np.array(features)  # shape: (num_frames, num_variants, 384)
def generate_image_variants(image: Image.Image) -> List[Image.Image]:
    """为一张图片生成变体：如果 ADVANCED_SPLIT 非空，则分割成块和组合网格；否则生成5个变体：原始 + 4个角落裁剪。"""
    if ADVANCED_SPLIT:
        parts = ADVANCED_SPLIT.split(',')
        base_split = parts[0]
        combinations = parts[1:] if len(parts) > 1 else []
        
        # 解析基础 h x w
        h, w = map(int, base_split.split('x'))
        width, height = image.size
        block_width = width // w
        block_height = height // h
        
        # 生成基础块
        blocks = {}
        variants = []
        for i in range(h):
            for j in range(w):
                left = j * block_width
                upper = i * block_height
                right = left + block_width
                lower = upper + block_height
                block = image.crop((left, upper, right, lower))
                block_id = f"{i}.{j}"
                blocks[block_id] = block
                variants.append(block)
        
        # 生成组合网格
        for combo in combinations:
            combo = combo.strip('[]')
            block_ids = combo.split('-')
            if len(block_ids) == 2:
                # 解析范围：start-end
                start_id = block_ids[0]
                end_id = block_ids[1]
                start_i, start_j = map(int, start_id.split('.'))
                end_i, end_j = map(int, end_id.split('.'))
                combo_blocks = [blocks[f"{i}.{j}"] for i in range(start_i, end_i + 1) for j in range(start_j, end_j + 1)]
            else:
                # 单个块或旧格式
                combo_blocks = [blocks[bid] for bid in block_ids if bid in blocks]
            if combo_blocks:
                # 假设组合是矩形网格，计算行数和列数
                num_blocks = len(combo_blocks)
                # 尝试找到合适的网格尺寸
                grid_h = int(num_blocks ** 0.5)
                grid_w = (num_blocks + grid_h - 1) // grid_h
                combined_image = combine_images_into_grid(combo_blocks, grid_h, grid_w)
                variants.append(combined_image)
        
        return variants
    else:
        # 原有逻辑：5个变体
        width, height = image.size
        # 裁剪大小为图片的SCALE_FACTOR倍
        crop_width = int(width * SCALE_FACTOR)
        crop_height = int(height * SCALE_FACTOR)
        variants = [image]  # 原始图片
        
        # 左上角
        variants.append(image.crop((0, 0, crop_width, crop_height)))
        # 右上角
        variants.append(image.crop((width - crop_width, 0, width, crop_height)))
        # 左下角
        variants.append(image.crop((0, height - crop_height, crop_width, height)))
        # 右下角
        variants.append(image.crop((width - crop_width, height - crop_height, width, height)))
        
        return variants


def combine_images_into_grid(images: List[Image.Image], rows: int, cols: int) -> Image.Image:
    """将图像列表拼接成网格图像。"""
    if not images:
        return Image.new('RGB', (1, 1))
    
    # 假设所有图像大小相同
    img_width, img_height = images[0].size
    grid_width = img_width * cols
    grid_height = img_height * rows
    
    grid_image = Image.new('RGB', (grid_width, grid_height))
    
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
        row = idx // cols
        col = idx % cols
        grid_image.paste(img, (col * img_width, row * img_height))
    
    return grid_image


def process_videos(video_paths: List[str], workdir: str, model_path: str):
    """处理视频，直接解压帧到内存并提取特征。"""
    os.makedirs(workdir, exist_ok=True)
    for video_path in video_paths:
        video_name = Path(video_path).stem
        print(f"Processing video: {video_name}")
        frames = extract_frames(video_path)
        features = extract_features_from_arrays(frames, model_path)
        feature_file = os.path.join(workdir, "video_features.npy")
        np.save(feature_file, features)
        print(f"Extracted features for `{video_name}` and saved to `{feature_file}`.")
        print(f"Features file size: {os.path.getsize(feature_file) / 1024:.2f} KB")
        num_variants = features.shape[1] if len(features.shape) > 1 else 1
        print(f"每个帧有 {num_variants} 个变体")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process-videos.py <video_path1> <video_path2> ...")
        sys.exit(1)

    video_paths = sys.argv[1:]
    workdir = "workdir"
    model_path = "models/dinov2-small/model.onnx"

    process_videos(video_paths, workdir, model_path)
