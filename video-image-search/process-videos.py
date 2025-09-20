"""
输入视频路径，通过 ffmpeg 提取视频帧，然后使用 CLIP 自动提取特征到 `workdir`。
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List
from PIL import Image
import numpy as np
import onnxruntime as ort


def extract_frames(video_path: str, fps: int = 1) -> List[np.ndarray]:
    """使用ffmpeg pipe直接读取视频帧到numpy数组，不经过磁盘。"""
    import math
    import re

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
    """预处理图片为CLIP模型输入，仿照官方_transform。"""
    # Resize + BICUBIC
    image = image.resize((target_size, target_size), Image.Resampling.BICUBIC)
    # 转为RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image).astype("float32") / 255.0
    # 官方归一化参数
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # HWC to CHW
    arr = np.expand_dims(arr, axis=0)  # Add batch dimension
    arr = arr.astype("float32")  # 保证为float32
    return arr


def extract_features_from_arrays(
    image_arrays: List[np.ndarray], model_path: str
) -> np.ndarray:
    """直接从帧数组提取特征。"""
    from tqdm import tqdm

    session = ort.InferenceSession(model_path)
    features = []
    for image_array in tqdm(image_arrays, desc="提取特征", ncols=80):
        image = load_image_from_array(image_array)
        input_tensor = preprocess_image(image)
        outputs = session.run(None, {"pixel_values": input_tensor})
        out = np.asarray(outputs[0])
        features.append(out[0])
    return np.array(features)


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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process-videos.py <video_path1> <video_path2> ...")
        sys.exit(1)

    video_paths = sys.argv[1:]
    workdir = "workdir"
    # model_path = "models/clip-ViT-B-32-multilingual-v1/model_quint8_avx2.onnx"
    model_path = "models/clip-ViT-B-32-vision/model.onnx"

    process_videos(video_paths, workdir, model_path)
