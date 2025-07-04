import numpy as np
import imageio
import scipy.ndimage
import cv2
import sys

def rgb2gray(rgb):
    """Convert RGB image to grayscale."""
    if len(rgb.shape) == 2:  # if already grayscale
        return rgb
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def dodge(front, back):
    """Apply the dodge technique to blend front and back images."""
    final_sketch = front * 255 / (255 - back + 1e-6)  # added small value to avoid division by zero
    final_sketch[final_sketch > 255] = 255
    final_sketch[back == 255] = 255
    return final_sketch.astype('uint8')

def convert_to_sketch(input_path, output_path):
    try:
        ss = imageio.imread(input_path)
        gray = rgb2gray(ss)
        i = 255 - gray
        blur = scipy.ndimage.gaussian_filter(i, sigma=15)
        r = dodge(blur, gray)
        cv2.imwrite(output_path, r)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sketch_converter.py <input_image> <output_image>")
    else:
        convert_to_sketch(sys.argv[1], sys.argv[2])