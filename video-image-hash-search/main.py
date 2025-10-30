import os
import sys
import glob
import cv2
import numpy as np
from PIL import ImageGrab
import datetime

def calculate_phash(frame, hash_size=16):
    """
    计算帧的感知哈希 (pHash) - 返回打包后的 uint8 数组
    """
    # 1. 灰度化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. 缩放: pHash 标准使用 32x32
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)

    # 3. DCT 变换 (使用浮点型)
    umat = cv2.UMat(np.float32(resized))
    dct_umat = cv2.dct(umat)
    dct = dct_umat.get()

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

def hamming_distance(hash1, hash2):
    """计算两个哈希的汉明距离"""
    # 1. 计算异或 (XOR)
    # 这会在两个哈希的比特位不同的地方标记为 1
    xor_result = np.bitwise_xor(hash1, hash2) # 结果还是 (32,) uint8

    # 2. 重新解包 (Unpack) 成比特位
    # (32,) 的 uint8 数组变回 (256,) 的 0/1 数组
    diff_bits = np.unpackbits(xor_result)

    # 3. 计算 1 的个数
    # 1 的个数就是它们之间不同的比特数，即汉明距离
    return np.sum(diff_bits)

def main(workdir: str, top_n: int = 20):
    # 从剪切板获取图片
    image = ImageGrab.grabclipboard()
    if image is None:
        print("剪切板中没有图片")
        sys.exit(1)

    # 转换为 cv2 格式
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 计算 query pHash
    query_phash = calculate_phash(frame)
    print(f"Query pHash shape: {query_phash.shape}, dtype: {query_phash.dtype}")

    # 读取所有 pHash 文件
    phash_files = glob.glob(os.path.join(workdir, "*_phashes.npz"))
    if not phash_files:
        print("没有找到 pHash 文件")
        sys.exit(1)

    results = []
    for file_path in phash_files:
        video_name = os.path.basename(file_path).replace("_phashes.npz", "")
        data = np.load(file_path)
        phash_array = data['phashes']
        fps = data['fps']
        for frame_idx, phash in enumerate(phash_array):
            dist = hamming_distance(query_phash, phash)
            time_sec = frame_idx / fps
            results.append((dist, video_name, frame_idx, time_sec))

    # 排序，取 top n
    results.sort(key=lambda x: x[0])
    top_results = results[:top_n]

    print(f"Top {top_n} matches:")
    for dist, video_name, frame_idx, time_sec in top_results:
        time_formatted = str(datetime.timedelta(seconds=time_sec)).split('.')[0]
        video_name_display = video_name if len(video_name) <= 20 else video_name[:17] + "..."
        print(f"距离: {dist}, Video: {video_name_display}, Frame: {frame_idx}, Time: {time_sec:.2f} s ({time_formatted})")

if __name__ == "__main__":
    workdir = "workdir"
    top_n = 20
    main(workdir, top_n)
