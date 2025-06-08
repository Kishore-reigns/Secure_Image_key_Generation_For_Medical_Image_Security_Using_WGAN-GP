import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import tkinter as tk
from tkinter import Button
from PIL import Image

ITERATIONS = 50

import numpy as np

def arnold_cat_map_forward(image):
    #image = image.resize((256, 256))
    image_np = np.array(image)  

    h, w, c = image_np.shape  
    transformed = image_np.copy()

    for _ in range(ITERATIONS):
        new_image = np.zeros_like(transformed)
        for i in range(c):  # Process each channel separately
            x, y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            new_x = (x + y) % h
            new_y = (x + 2 * y) % w
            new_image[new_x, new_y, i] = transformed[x, y, i]

        transformed = new_image.copy()

    transformed_image = Image.fromarray(transformed.astype(np.uint8), mode="RGB")
    #transformed_image.show()
    return transformed_image

def arnold_cat_map_forward__(image,iterations):
    #image = image.resize((256, 256))
    print(iterations)
    image_np = np.array(image)  

    h, w, c = image_np.shape  
    transformed = image_np.copy()

    for _ in range(iterations):
        new_image = np.zeros_like(transformed)
        for i in range(c):  # Process each channel separately
            x, y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            new_x = (x + y) % h
            new_y = (x + 2 * y) % w
            new_image[new_x, new_y, i] = transformed[x, y, i]

        transformed = new_image.copy()

    transformed_image = Image.fromarray(transformed.astype(np.uint8), mode="RGB")
    #transformed_image.show()
    return transformed_image


def arnold_cat_map_reverse(image):
    #image = image.resize((256, 256)).convert("RGB")  
    image_np = np.array(image)  

    h, w, c = image_np.shape
    transformed = image_np.copy()

    for _ in range(ITERATIONS):
        new_image = np.zeros_like(transformed)

        for i in range(c): 
            x, y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            new_x = ((2 * x - y) % h)
            new_y = ((-x + y) % w)
            new_image[new_x, new_y, i] = transformed[x, y, i]

        transformed = new_image.copy()

    transformed_image = Image.fromarray(transformed.astype(np.uint8), mode="RGB")

    #transformed_image.show()
    return transformed_image

def XOR(image,key):
    encrypted_array = np.bitwise_xor(np.array(image), np.array(key))
    image = Image.fromarray(encrypted_array)
    #image.show()
    return image


def encrypt(image,key):
    scambled_image = arnold_cat_map_forward(image)
    encrypted_image = XOR(scambled_image,key)
    return encrypted_image

def decrypt(image,key):
    
    reverse_xor = XOR(image,key)
    decrypted_image = arnold_cat_map_reverse(reverse_xor)
    return decrypted_image


if __name__ == '__main__':
    image_path = "./mri_images/MCUCXR_0108_1.png"
    image = Image.open(image_path)
    #image_np = np.array(image) 

    test_key = Image.open("./image-key/image_key_pair_19/key.jpg")
    
    image.show()
    test_key.show()

    scrambled_image = arnold_cat_map_forward(image)
    scrambled_image.show()
    encrypted_image = XOR(scrambled_image,test_key)
    encrypted_image.show()

    decrypted_image = XOR(encrypted_image,test_key)
    unScrambled_image = arnold_cat_map_reverse(decrypted_image)

    # Display Original and Scrambled Images Side by Side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(scrambled_image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    ax[1].imshow(unScrambled_image)
    ax[1].set_title(f"Uncrambled Image after {ITERATIONS} Iterations")
    ax[1].axis("off")
    plt.show()
