import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict
import cv2
import numpy as np
import queue
import threading

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

def frame_reader(proc, frame_size, height, width, frame_queues):
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            for q in frame_queues:
                q.put(frame)
    finally:
        proc.stdout.close()
        proc.wait()
    for q in frame_queues:
        q.put(None)

def phash_worker(hs, frame_queue, phashes):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        phash = calculate_phash(frame, hs)
        phashes[hs].append(phash)

def extract_phashes(video_path: str, hash_sizes: List[int], fps: float = 0, use_ffmpeg_resize: bool = False) -> tuple[Dict[int, List[np.ndarray]], float]:
    """使用ffmpeg pipe流式读取视频帧，计算pHash，不保存原始帧到内存。fps=0 表示使用视频原始帧率。use_ffmpeg_resize=True 时，ffmpeg 输出 32x32 帧；False 时，输出原始分辨率帧。"""

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

    if use_ffmpeg_resize:
        output_width, output_height = 32, 32
        vf_filter = f"fps={actual_fps},scale={output_width}:{output_height}"
    else:
        output_width, output_height = width, height
        vf_filter = f"fps={actual_fps}"

    # 用ffmpeg pipe输出raw BGR帧
    ffmpeg_cmd = [
        "ffmpeg",
        "-hwaccel",
        "auto",
        "-i",
        video_path,
        "-vf",
        vf_filter,
        "-f",
        "image2pipe",
        "-pix_fmt",
        "bgr24",
        "-vcodec",
        "rawvideo",
        "-",
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)
    frame_size = output_width * output_height * 3
    frame_queues = [queue.Queue(maxsize=10) for _ in hash_sizes]  # one queue per worker
    phashes = {hs: [] for hs in hash_sizes}

    # start reader thread
    reader_thread = threading.Thread(target=frame_reader, args=(proc, frame_size, output_height, output_width, frame_queues))
    reader_thread.start()

    # start worker threads
    worker_threads = []
    for i, hs in enumerate(hash_sizes):
        t = threading.Thread(target=phash_worker, args=(hs, frame_queues[i], phashes))
        t.start()
        worker_threads.append(t)

    # wait for all threads
    reader_thread.join()
    for t in worker_threads:
        t.join()

    print(
        f"Extracted and computed pHashes for hash_sizes {hash_sizes} from `{video_path}` at {actual_fps} fps (ffmpeg_resize={use_ffmpeg_resize})."
    )
    return phashes, actual_fps

def process_videos(video_paths: List[str], workdir: str):
    """处理视频，提取帧，计算pHash，保存到本地文件。"""
    os.makedirs(workdir, exist_ok=True)
    hash_sizes = [8, 12, 16]
    for video_path in video_paths:
        video_name = Path(video_path).stem
        print(f"Processing video: {video_name}")
        phashes_dict, fps = extract_phashes(video_path, hash_sizes, use_ffmpeg_resize=True)
        phashes_data = {f'phashes_{hs}': np.array(phashes_dict[hs], dtype=np.uint8) for hs in hash_sizes}
        feature_file = os.path.join(workdir, f"{video_name}_phashes.npz")
        np.savez(feature_file, **phashes_data, fps=fps)
        print(f"Saved pHashes and fps for `{video_name}` to `{feature_file}`.")
        print(f"Number of frames: {len(phashes_dict[hash_sizes[0]])}, FPS: {fps}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process-videos.py <video_path1> <video_path2> ...")
        sys.exit(1)

    video_paths = sys.argv[1:]

    if not video_paths:
        print("No video paths provided")
        sys.exit(1)

    workdir = "workdir"
    process_videos(video_paths, workdir)
