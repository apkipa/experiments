import os
import subprocess
import sys
from pathlib import Path
from typing import List
import cv2
import numpy as np

def calculate_phash(frame, hash_size=16):
    """
    计算帧的感知哈希 (pHash) - 返回打包后的 uint8 数组
    """
    # 1. 灰度化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. 缩放: pHash 标准使用 32x32
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)

    # 3. DCT 变换 (使用浮点型)
    dct = cv2.dct(np.float32(resized))

    # 4. 取左上角 hash_size x hash_size 低频区域
    dct_roi = dct[0:hash_size, 0:hash_size]

    # 5. 计算中位数
    median_val = np.median(dct_roi)

    # 6. 生成哈希位: 大于中位数为 1，否则为 0
    hash_bits = (dct_roi > median_val) # 形状 (hash_size, hash_size)

    # 7. 【修改】打包成 uint8 数组
    # 首先将 (16, 16) 的布尔值扁平化为 (256,)
    # 然后 np.packbits 将其打包为 (32,) 的 uint8 数组
    return np.packbits(hash_bits.flatten())

def extract_phashes(video_path: str, fps: float = 0, hash_size=16) -> tuple[List[np.ndarray], float]:
    """使用ffmpeg pipe流式读取视频帧，计算pHash，不保存原始帧到内存。fps=0 表示使用视频原始帧率。"""

    # 获取视频宽高和帧率
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "csv=p=0",
        video_path,
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe.stderr}")
    parts = probe.stdout.strip().split(",")
    width, height = int(parts[0]), int(parts[1])
    fps_str = parts[2]
    num, den = fps_str.split('/')
    video_fps = float(num) / float(den)

    if fps == 0:
        actual_fps = video_fps
    else:
        actual_fps = fps

    # 用ffmpeg pipe输出raw RGB帧
    ffmpeg_cmd = [
        "ffmpeg",
        "-hwaccel",
        "auto",
        "-i",
        video_path,
        "-vf",
        f"fps={actual_fps}",
        "-f",
        "image2pipe",
        "-pix_fmt",
        "rgb24",
        "-vcodec",
        "rawvideo",
        "-",
    ]
    # proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    # Showing stderr to user may help debug ffmpeg issues
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)
    frame_size = width * height * 3
    phashes = []
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
        phash = calculate_phash(frame, hash_size)
        phashes.append(phash)
    proc.stdout.close()
    proc.wait()
    print(
        f"Extracted and computed {len(phashes)} pHashes from `{video_path}` at {actual_fps} fps."
    )
    return phashes, actual_fps

def process_videos(video_paths: List[str], workdir: str, hash_size=16):
    """处理视频，提取帧，计算pHash，保存到本地文件。"""
    os.makedirs(workdir, exist_ok=True)
    for video_path in video_paths:
        video_name = Path(video_path).stem
        print(f"Processing video: {video_name}")
        phashes, fps = extract_phashes(video_path, hash_size=hash_size)
        
        # 【修改】
        # phashes 现在是一个列表，每个元素是 (32,) 的 uint8 数组
        # np.array(phashes) 会自动创建一个 (N_frames, 32) 的 uint8 数组
        # 原代码: phash_array = np.array(phashes, dtype=np.uint64)
        phash_array = np.array(phashes, dtype=np.uint8) # 显式指定 dtype
        feature_file = os.path.join(workdir, f"{video_name}_phashes.npz")
        np.savez(feature_file, phashes=phash_array, fps=fps)
        print(f"Saved pHashes and fps for `{video_name}` to `{feature_file}`.")
        print(f"Number of frames: {len(phashes)}, FPS: {fps}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process-videos.py <video_path1> <video_path2> ... [--hash_size <size>]")
        sys.exit(1)

    video_paths = []
    hash_size = 16
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--hash_size':
            if i + 1 < len(sys.argv):
                hash_size = int(sys.argv[i + 1])
                i += 2
            else:
                print("Error: --hash_size requires a value")
                sys.exit(1)
        else:
            video_paths.append(sys.argv[i])
            i += 1

    if not video_paths:
        print("No video paths provided")
        sys.exit(1)

    print("Using pHash size =", hash_size)

    workdir = "workdir"
    process_videos(video_paths, workdir, hash_size)
