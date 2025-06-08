import numpy as np 
from PIL import Image
import cryptoSystem as cs
import random


def add_white_noise(pil_img, noise_level=200):
    # Convert PIL image to numpy array
    img_array = np.array(pil_img).astype(np.int16)  # to prevent overflow

    # Generate random noise
    noise = np.random.randint(-noise_level, noise_level + 1, img_array.shape)

    # Add noise and clip the values to [0, 255]
    noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    # Convert back to PIL
    noisy_pil_img = Image.fromarray(noisy_img_array)
    return noisy_pil_img



def key_metrics(key):
    key = cs.arnold_cat_map_forward__(key,random.randint(1,192))
    key = add_white_noise(key,200)
    return key
    