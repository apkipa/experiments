import os
import cv2
import numpy as np
import argparse
import sys
import ctypes

# Enable DPI awareness for proper display on high-DPI screens
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except AttributeError:
    pass  # Not on Windows or shcore not available

def main():
    parser = argparse.ArgumentParser(description='Detect ORB features in an image.')
    parser.add_argument('image_path', nargs='?', default=None, help='Path to the input image (optional, if not provided, read from clipboard)')
    parser.add_argument('n_features', nargs='?', type=int, default=200, help='Number of ORB features to detect (default: 200)')
    args = parser.parse_args()

    # Read image
    if args.image_path is None:
        try:
            import PIL.ImageGrab
            pil_img = PIL.ImageGrab.grabclipboard()
            if pil_img is None:
                print("No image found in clipboard.")
                sys.exit(1)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except ImportError:
            print("PIL (Pillow) is required to read from clipboard. Install with: pip install pillow")
            sys.exit(1)
    else:
        img = cv2.imread(args.image_path)
        if img is None:
            print(f"Error: Cannot read image from {args.image_path}")
            sys.exit(1)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=args.n_features)

    # Detect keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(img, None)

    if descriptors is None:
        print("No features detected.")
        sys.exit(1)

    # Draw keypoints on the image
    img_with_keypoints = img.copy()
    cv2.drawKeypoints(img, keypoints, img_with_keypoints, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Resize image for better display
    max_width = 800
    max_height = 600
    h, w = img_with_keypoints.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1.0 or (w < 200 or h < 200):  # 如果太大或太小，缩放
        scale = max(scale, 0.5) if scale < 1.0 else min(scale, 2.0)  # 限制缩放范围
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_with_keypoints = cv2.resize(img_with_keypoints, (new_w, new_h))

    # Display in resizable window
    cv2.namedWindow('ORB Features', cv2.WINDOW_NORMAL)
    cv2.imshow('ORB Features', img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save descriptors to binary file
    descriptors.astype(np.uint8).tofile('orb_features.bin')
    print("ORB features saved to orb_features.bin")

    # Print file size
    file_size = os.path.getsize('orb_features.bin')
    print(f"File size: {file_size / 1024:.2f} KB")

if __name__ == "__main__":
    main()
