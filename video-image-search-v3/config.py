# Configuration file for video-image-search-v2

# Scale factor for cropping image variants (0.5 means 50% of width/height)
SCALE_FACTOR = 0.7

# Whether to use GPU acceleration for ONNX inference
USE_GPU = False

# Quantization type: 'none', 'float16', 'bfloat16', 'float8_e4m3', 'int8'
QUANT_TYPE = 'none'

# Advanced split strategy for video frames: "hxw,[block1-block2-...],[...]" format, e.g., "2x3,[0.0-1.1],[0.1-1.2]" for 2 rows and 3 columns plus combined grids
ADVANCED_SPLIT = "2x3,[0.0-1.1],[0.1-1.2],[0.0-1.2]"
