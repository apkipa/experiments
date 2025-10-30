import cv2
import numpy as np
import sys

# def calculate_phash(frame, hash_size=8):
#     """
#     计算帧的感知哈希 (pHash) - 使用标准的中位数方法
#     """
#     # 1. 灰度化
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # 2. 缩放: pHash 标准使用 32x32
#     resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    
#     # 3. DCT 变换 (使用浮点型)
#     umat = cv2.UMat(np.float32(resized))
#     dct_umat = cv2.dct(umat)
#     dct = dct_umat.get()
    
#     # 4. 取左上角 8x8 低频区域
#     dct_roi = dct[0:hash_size, 0:hash_size]
    
#     # 5. 计算中位数 (!!! 这是关键修正 !!!)
#     median_val = np.median(dct_roi)
    
#     # 6. 生成哈希位: 大于中位数为 1，否则为 0
#     hash_bits = (dct_roi > median_val)
    
#     # 7. 位打包 (Bit-packing)
#     hash_int = 0
#     bit_length = hash_size * hash_size
#     for i in range(hash_size):
#         for j in range(hash_size):
#             if hash_bits[i, j]:
#                 # 按行优先顺序打包
#                 hash_int |= (1 << (bit_length - 1 - (i * hash_size + j)))
                
#     # 8. 转换为十六进制字符串
#     hex_length = (bit_length + 3) // 4
#     return f'{hash_int:0{hex_length}x}'

import cv2
import numpy as np

def calculate_phash(frame, hash_size=8):
    """
    计算帧的感知哈希 (pHash) - 使用标准的中位数方法
    """
    # 1. 灰度化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. 缩放: pHash 标准使用 32x32
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    
    # 3. DCT 变换 (使用浮点型)
    #    (使用 UMat 是一个好优化，但为清晰起见，这里用标准 numpy)
    dct = cv2.dct(np.float32(resized))
    
    # 4. 取左上角 8x8 低频区域
    dct_roi = dct[0:hash_size, 0:hash_size]
    
    # 5. 计算中位数 (!!! 这是关键修正 !!!)
    median_val = np.median(dct_roi)
    
    # 6. 生成哈希位: 大于中位数为 1，否则为 0
    hash_bits = (dct_roi > median_val)
    
    # 7. 位打包 (Bit-packing)
    hash_int = 0
    bit_length = hash_size * hash_size
    for i in range(hash_size):
        for j in range(hash_size):
            if hash_bits[i, j]:
                # 按行优先顺序打包
                hash_int |= (1 << (bit_length - 1 - (i * hash_size + j)))
                
    # 8. 转换为十六进制字符串
    hex_length = (bit_length + 3) // 4
    return f'{hash_int:0{hex_length}x}'

def main(video_path, hash_size=8):
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 尝试启用硬件加速
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    
    # 检查硬件加速状态
    hw_accel = cap.get(cv2.CAP_PROP_HW_ACCELERATION)
    if hw_accel > 0:
        print(f"硬件加速已启用: {hw_accel}")
    else:
        print("硬件加速未启用，使用软件解码")

    # 获取视频属性
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建窗口
    window_name = "Video Player with pHash"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # 调整窗口大小，使其更小并可调整
    cv2.resizeWindow(window_name, width // 2, (height // 2) + 114)

    # 创建trackbar用于seek
    def on_trackbar(pos):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    cv2.createTrackbar('Frame', window_name, 0, total_frames - 1, on_trackbar)

    # 播放循环
    paused = False
    current_frame = 0
    # hash_size 从参数传入
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # 因为read后指针已前进

        # 计算pHash
        phash_value = calculate_phash(frame, hash_size)
        # 将pHash转换为二进制字符串
        bit_length = hash_size * hash_size
        phash_binary = bin(int(phash_value, 16))[2:].zfill(bit_length)

        # 创建hash_size x hash_size的黑白图像表示pHash
        hash_image = np.zeros((hash_size, hash_size), dtype=np.uint8)
        for i in range(hash_size):
            for j in range(hash_size):
                bit = int(phash_binary[i * hash_size + j])
                hash_image[i, j] = 255 if bit else 0
        # 放大到64x64
        hash_display = cv2.resize(hash_image, (64, 64), interpolation=cv2.INTER_NEAREST)
        # 转换为BGR
        hash_display_bgr = cv2.cvtColor(hash_display, cv2.COLOR_GRAY2BGR)

        # 缩放帧以适应窗口
        frame_small = cv2.resize(frame, (width // 2, height // 2))

        # 创建显示图像，增加下方空间
        display_height = (height // 2) + 114
        display_image = np.zeros((display_height, width // 2, 3), dtype=np.uint8)
        display_image[:height // 2, :] = frame_small

        # 将pHash图像放置在右下角
        display_image[(height // 2) + 50 : (height // 2) + 114, (width // 2) - 64 : width // 2] = hash_display_bgr

        # 显示帧号
        cv2.putText(display_image, f"Frame: {current_frame}", (10, (height // 2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 显示pHash（十六进制）
        cv2.putText(display_image, f"pHash: {phash_value}", (10, (height // 2) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 显示当前哈希大小
        cv2.putText(display_image, f"Hash Size: {hash_size}x{hash_size}", (10, (height // 2) + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 显示图像
        cv2.imshow(window_name, display_image)

        # 更新trackbar
        cv2.setTrackbarPos('Frame', window_name, current_frame)

        # 键盘控制
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # 空格键暂停/播放
            paused = not paused
        elif key == ord('+'):  # 增加哈希大小
            hash_size = min(hash_size + 1, 16)
        elif key == ord('-'):  # 减少哈希大小
            hash_size = max(hash_size - 1, 4)
        elif key == ord('r'):  # 重置到开头
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_frame = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python main.py <video_path> [hash_size]")
        print("hash_size 默认为8，可选范围4-16")
        sys.exit(1)
    video_path = sys.argv[1]
    hash_size = int(sys.argv[2]) if len(sys.argv) >= 3 else 8
    # hash_size = max(4, min(16, hash_size))  # 限制在4-16
    main(video_path, hash_size)
