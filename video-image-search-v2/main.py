import os
import requests
from tqdm import tqdm
import numpy as np
from PIL import Image
import onnxruntime as ort
import sys
import time
from tokenizers import Tokenizer
from safetensors.numpy import load_file  # 需要 pip install safetensors
import ml_dtypes
from typing import List
from config import SCALE_FACTOR, QUANT_TYPE, ADVANCED_SPLIT


def download_file(url: str, save_path: str):
    """Download a file from a URL to a specified local path."""

    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we notice bad responses

    total = int(response.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with (
        open(save_path, "wb") as file,
        tqdm(
            desc=f"Downloading {os.path.basename(save_path)}",
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            bar.update(size)

    print(f"Downloaded `{url}` to `{save_path}`.")


def download_file_if_not_exists(url: str, save_path: str):
    """Download a file only if it does not already exist."""
    if not os.path.exists(save_path):
        print(f"File `{save_path}` does not exist. Downloading...")
        download_file(url, save_path)
    else:
        # print(f"File {save_path} already exists. Skipping download.")
        pass


def download_models():
    """Download required model files if they do not exist."""
    # download_file_if_not_exists(
    #     "https://hf-mirror.com/sentence-transformers/clip-ViT-B-32-multilingual-v1/resolve/main/onnx/model_quint8_avx2.onnx",
    #     "models/clip-ViT-B-32-multilingual-v1/model_quint8_avx2.onnx",
    # )
    download_file_if_not_exists(
        "https://hf-mirror.com/sentence-transformers/clip-ViT-B-32-multilingual-v1/resolve/main/onnx/model_O4.onnx",
        "models/clip-ViT-B-32-multilingual-v1/model_O4.onnx",
    )
    download_file_if_not_exists(
        "https://hf-mirror.com/sentence-transformers/clip-ViT-B-32-multilingual-v1/resolve/main/2_Dense/model.safetensors",
        "models/clip-ViT-B-32-multilingual-v1/2_Dense/model.safetensors",
    )
    download_file_if_not_exists(
        "https://hf-mirror.com/sentence-transformers/clip-ViT-B-32-multilingual-v1/resolve/main/tokenizer.json",
        "models/clip-ViT-B-32-multilingual-v1/tokenizer.json",
    )
    # download_file_if_not_exists(
    #     "https://hf-mirror.com/sentence-transformers/clip-ViT-B-32-multilingual-v1/resolve/main/tokenizer_config.json",
    #     "models/clip-ViT-B-32-multilingual-v1/tokenizer_config.json",
    # )
    # download_file_if_not_exists(
    #     "https://hf-mirror.com/sentence-transformers/clip-ViT-B-32-multilingual-v1/resolve/main/vocab.txt",
    #     "models/clip-ViT-B-32-multilingual-v1/vocab.txt",
    # )
    # download_file_if_not_exists(
    #     "https://hf-mirror.com/sentence-transformers/clip-ViT-B-32-multilingual-v1/resolve/main/special_tokens_map.json",
    #     "models/clip-ViT-B-32-multilingual-v1/special_tokens_map.json",
    # )
    download_file_if_not_exists(
        "https://hf-mirror.com/Qdrant/clip-ViT-B-32-vision/resolve/main/model.onnx",
        "models/clip-ViT-B-32-vision/model.onnx",
    )


# def preprocess_image(image: Image.Image, target_size: int = 224) -> np.ndarray:
#     """预处理图片为CLIP模型输入。"""
#     image = image.resize((target_size, target_size))
#     arr = np.array(image).astype("float32") / 255.0
#     arr = (arr - 0.48145466) / 0.26862954  # Normalize
#     arr = np.transpose(arr, (2, 0, 1))  # HWC to CHW
#     arr = np.expand_dims(arr, axis=0)  # Add batch dimension
#     return arr


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


def generate_image_variants(image: Image.Image) -> List[Image.Image]:
    """为一张图片生成5个变体：原始 + 4个角落裁剪。"""
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


def extract_feature(image: Image.Image, model_path: str) -> np.ndarray:
    start_time = time.time()
    print("开始图片特征提取")

    # 图像预处理
    preprocess_start = time.time()
    input_tensor = preprocess_image(image)
    preprocess_time = time.time() - preprocess_start
    print(f"  ├─ 图像预处理耗时: {preprocess_time:.3f}秒")

    # 模型推理
    inference_start = time.time()
    session = ort.InferenceSession(model_path)
    outputs = session.run(None, {"pixel_values": input_tensor})
    inference_time = time.time() - inference_start
    print(f"  ├─ 图像模型推理耗时: {inference_time:.3f}秒")

    total_time = time.time() - start_time
    print(f"  └─ 图片特征提取总耗时: {total_time:.3f}秒")

    return outputs[0][0]


def get_image(path: str) -> Image.Image:
    if path:
        return Image.open(path)
    else:
        # 从剪贴板读取图片
        try:
            from PIL import ImageGrab

            img = ImageGrab.grabclipboard()
            if isinstance(img, Image.Image):
                return img
            else:
                print("剪贴板无图片！")
                sys.exit(1)
        except Exception as e:
            print(f"剪贴板读取失败: {e}")
            sys.exit(1)


def make_tokenizer(model_dir: str):
    """创建并配置CLIP tokenizer。"""
    tokenizer_file_path = os.path.join(model_dir, "tokenizer.json")

    try:
        tokenizer = Tokenizer.from_file(tokenizer_file_path)
        print("✅ Tokenizer loaded successfully using the `tokenizers` library!")
    except Exception as e:
        print(
            f"❌ Failed to load '{tokenizer_file_path}'. Please ensure the file exists and is valid."
        )
        print(f"Error details: {e}")
        sys.exit(1)

    # 配置Padding
    pad_token = "[PAD]"
    pad_token_id = tokenizer.token_to_id(pad_token)
    if pad_token_id is not None:
        tokenizer.enable_padding(pad_id=pad_token_id, pad_token=pad_token)
        print(f"Padding enabled. PAD token ID: {pad_token_id}")
    else:
        print("Warning: [PAD] token not found.")

    # 配置截断
    tokenizer.enable_truncation(max_length=77)

    return tokenizer


# def extract_text_feature(text: str, model_dir: str) -> np.ndarray:
#     """
#     使用多语言CLIP模型提取文本特征。
#     此版本会检查是否存在偏置项（bias），并自适应处理。
#     """
#     # 路径定义
#     model_path = os.path.join(model_dir, "model_O4.onnx")
#     dense_path = os.path.join(model_dir, "2_Dense/model.safetensors")

#     # 1. Tokenization
#     tokenizer = make_tokenizer(model_dir)
#     encodings = tokenizer.encode_batch([text])

#     input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
#     attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)

#     # 2. ONNX 推理 (获取 768 维词向量)
#     session = ort.InferenceSession(model_path)
#     outputs = session.run(None, {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask
#     })
#     token_embeddings = outputs[0]  # shape: (1, 77, 768)

#     # 3. Mean Pooling (计算平均池化)
#     mask_expanded = np.expand_dims(attention_mask, axis=-1).repeat(token_embeddings.shape[-1], axis=-1)
#     sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
#     sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
#     pooled_feat_768 = sum_embeddings / sum_mask # shape: (1, 768)

#     # 4. Dense Layer (应用全连接层)
#     dense_tensors = load_file(dense_path)
#     W = dense_tensors["linear.weight"]  # shape: (512, 768)

#     # 核心修正：只进行矩阵乘法
#     text_feat_512 = np.dot(pooled_feat_768, W.T)

#     # 健壮性检查：如果文件中存在 bias，则加上它
#     if "linear.bias" in dense_tensors:
#         print("发现并应用偏置项 (bias)。")
#         b = dense_tensors["linear.bias"]
#         text_feat_512 += b

#     return text_feat_512[0] # 返回 shape: (512,) 的向量

# def extract_text_feature(text: str, model_dir: str) -> np.ndarray:
#     """
#     使用多语言CLIP模型提取文本特征。
#     此版本将池化策略从 Mean Pooling 修改为 CLS Pooling。
#     """
#     # 路径定义
#     model_path = os.path.join(model_dir, "model_O4.onnx")
#     dense_path = os.path.join(model_dir, "2_Dense/model.safetensors")

#     # 1. Tokenization
#     tokenizer = make_tokenizer(model_dir)
#     encodings = tokenizer.encode_batch([text])

#     input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
#     attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)

#     # 2. ONNX 推理 (获取 768 维词向量)
#     session = ort.InferenceSession(model_path)
#     outputs = session.run(None, {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask
#     })
#     token_embeddings = outputs[0]  # shape: (1, 77, 768)

#     # ------------------- 核心修改点 -------------------
#     # 3. CLS Pooling (获取 [CLS] token 的向量)
#     # [CLS] token 是序列的第一个 token。
#     # token_embeddings[0] 选择第一个（也是唯一一个）句子。
#     # [0] 选择该句子的第一个 token。
#     pooled_feat_768 = token_embeddings[0][0] # shape: (768,)
#     # ---------------------------------------------------

#     # 4. Dense Layer (应用全连接层)
#     dense_tensors = load_file(dense_path)
#     W = dense_tensors["linear.weight"]  # shape: (512, 768)

#     # 将 pooled_feat_768 变回 2D array 以便进行矩阵乘法
#     # np.dot((768,), (768, 512)) -> (512,)
#     text_feat_512 = np.dot(pooled_feat_768, W.T)

#     # 检查并应用偏置项（虽然此模型没有，但这是健壮的写法）
#     if "linear.bias" in dense_tensors:
#         b = dense_tensors["linear.bias"]
#         text_feat_512 += b

#     # 注意：text_feat_512 此处已经是 (512,) 的一维向量，无需再取[0]
#     return text_feat_512


def extract_text_feature(text: str, model_dir: str) -> np.ndarray:
    """
    最终修正版：
    1. 使用配置文件明确规定的 Mean Pooling 策略。
    2. 增加了一个在池化后、Dense层前的 L2 Normalization 步骤。
    """
    start_time = time.time()
    print(f"开始文本特征提取: '{text}'")

    # ... (前面的路径定义、Tokenization、ONNX推理部分保持不变) ...
    # ... (确保 tokenizer 和 onnx session 的加载是正确的) ...
    model_path = os.path.join(model_dir, "model_O4.onnx")
    # model_path = os.path.join(model_dir, "model_quint8_avx2.onnx")
    dense_path = os.path.join(model_dir, "2_Dense/model.safetensors")

    # 1. Tokenization
    tokenization_start = time.time()
    tokenizer = make_tokenizer(model_dir)
    encodings = tokenizer.encode_batch([text])
    input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)
    tokenization_time = time.time() - tokenization_start
    print(f"  ├─ 文本分词耗时: {tokenization_time:.3f}秒")

    # 2. ONNX 推理
    inference_start = time.time()
    session = ort.InferenceSession(model_path)
    outputs = session.run(
        None, {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    token_embeddings = outputs[0]  # shape: (1, 77, 768)
    inference_time = time.time() - inference_start
    print(f"  ├─ ONNX模型推理耗时: {inference_time:.3f}秒")

    # 3. Mean Pooling (根据模型配置，这是唯一正确的策略)
    pooling_start = time.time()
    mask_expanded = np.expand_dims(attention_mask, axis=-1).repeat(
        token_embeddings.shape[-1], axis=-1
    )
    sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
    sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    pooled_feat_768 = sum_embeddings / sum_mask  # shape: (1, 768)

    # ------------------- 增加的关键步骤 -------------------
    # 4. L2 Normalization (在送入Dense层前进行归一化)
    # 这是CLIP架构中的一个常见操作，很可能是被我们忽略的关键
    norm = np.linalg.norm(pooled_feat_768, axis=1, keepdims=True)
    pooled_feat_768_normalized = pooled_feat_768 / norm
    # ---------------------------------------------------

    # 5. Dense Layer (使用归一化后的特征)
    dense_tensors = load_file(dense_path)
    W = dense_tensors["linear.weight"]

    # 使用归一化后的pooled_feat_768_normalized进行计算
    text_feat_512 = np.dot(pooled_feat_768_normalized, W.T)

    if "linear.bias" in dense_tensors:  # 对于此模型，这部分不会执行
        b = dense_tensors["linear.bias"]
        text_feat_512 += b

    pooling_time = time.time() - pooling_start
    print(f"  ├─ 特征池化和投影耗时: {pooling_time:.3f}秒")

    total_time = time.time() - start_time
    print(f"  └─ 文本特征提取总耗时: {total_time:.3f}秒")

    return text_feat_512[0]


def quantize_features(features: np.ndarray, quant_type: str):
    """对特征进行量化，返回量化后的特征（已反量化回 float32）。"""
    if quant_type == 'none':
        return features
    elif quant_type == 'float16':
        return features.astype(np.float16).astype(np.float32)
    elif quant_type == 'bfloat16':
        return features.astype(ml_dtypes.bfloat16).astype(np.float32)
    elif quant_type == 'float8_e4m3':
        return features.astype(ml_dtypes.float8_e4m3).astype(np.float32)
    elif quant_type == 'int8':
        # 对称量化，假设范围 [-1, 1]
        scale = 127.0  # max_abs / 127, max_abs=1
        quantized = np.round(features / (1.0 / scale)).astype(np.int8)
        # 立即反量化
        dequantized = quantized.astype(np.float32) * (1.0 / scale)
        return dequantized
    else:
        raise ValueError(f"Unsupported quant_type: {quant_type}")


if __name__ == "__main__":
    # overall_start = time.time()

    # 模型下载阶段
    download_start = time.time()
    download_models()
    download_time = time.time() - download_start
    print(f"模型下载检查耗时: {download_time:.3f}秒")
    print()

    user_input = input("输入文本内容，或留空从剪切板获取图片：")

    # 加载视频帧特征
    load_start = time.time()
    print("正在加载视频帧特征...")
    video_feats = np.load("workdir/video_features.npy")  # shape: (num_frames, num_variants, 512)
    num_frames = video_feats.shape[0]
    num_variants = video_feats.shape[1]
    video_feats = video_feats / np.linalg.norm(video_feats, axis=2, keepdims=True)
    
    # 量化
    video_feats = quantize_features(video_feats, QUANT_TYPE)
    
    load_time = time.time() - load_start
    print(f"视频特征加载耗时: {load_time:.3f}秒")
    print(f"总帧数: {num_frames}，每帧{num_variants}个向量，量化类型: {QUANT_TYPE}")

    # 查询特征提取阶段
    if user_input.strip() == "":
        # 处理图片
        img = get_image("")
        model_path = "models/clip-ViT-B-32-vision/model.onnx"
        query_feat = extract_feature(img, model_path)
    else:
        # 处理文本
        model_dir = "models/clip-ViT-B-32-multilingual-v1"
        query_feat = extract_text_feature(user_input, model_dir)

    # 后处理阶段
    postprocess_start = time.time()
    print("开始后处理...")

    # 归一化
    normalization_start = time.time()
    query_feat = query_feat / np.linalg.norm(query_feat)
    normalization_time = time.time() - normalization_start
    print(f"  ├─ 查询特征归一化耗时: {normalization_time:.3f}秒")

    # 计算相似度（cosine similarity）
    similarity_start = time.time()
    # video_feats: (num_frames, num_variants, 512), query_feat: (512,)
    sims = np.dot(video_feats, query_feat)  # (num_frames, num_variants)
    sims_flat = sims.flatten()  # (num_frames * num_variants,)
    similarity_time = time.time() - similarity_start
    print(f"  ├─ 相似度计算耗时: {similarity_time:.3f}秒")

    # 排序和选择Top-K
    topk_start = time.time()
    top_k = 20
    idxs_flat = np.argsort(sims_flat)[::-1][:top_k]
    # 转换回帧号和向量号
    frame_indices = idxs_flat // num_variants
    variant_indices = idxs_flat % num_variants
    topk_time = time.time() - topk_start
    print(f"  ├─ Top-K排序耗时: {topk_time:.3f}秒")

    postprocess_time = time.time() - postprocess_start
    print(f"  └─ 后处理总耗时: {postprocess_time:.3f}秒")

    overall_time = time.time() - load_start
    print(f"\n🎯 整体推理流程总耗时: {overall_time:.3f}秒")
    print("=" * 50)

    print("\nTop K 相似帧:")
    if ADVANCED_SPLIT:
        parts = ADVANCED_SPLIT.split(',')
        base_split = parts[0]
        combinations = parts[1:] if len(parts) > 1 else []
        h, w = map(int, base_split.split('x'))
        variant_names = [f"{i}.{j}" for i in range(h) for j in range(w)]
        for combo in combinations:
            combo = combo.strip('[]')
            variant_names.append(f"组合[{combo}]")
    else:
        variant_names = ["原始", "左上角", "右上角", "左下角", "右下角"]
    for i, (frame_idx, variant_idx) in enumerate(zip(frame_indices, variant_indices)):
        sim = sims_flat[idxs_flat[i]]
        print(f"第{frame_idx}帧 ({frame_idx // 60:02d}分{frame_idx % 60:02d}秒) - {variant_names[variant_idx]}，相似度: {sim:.4f}")
