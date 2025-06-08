from PIL import Image
import os

# Function to convert image to binary sequence
def image_to_binary(image_path):
    with Image.open(image_path) as img:
        img = img.convert('1')  # Convert image to black and white (1-bit pixels)
        pixels = list(img.getdata())  # Get pixel data (0 or 255 for black and white)
        binary_seq = ''.join(['0' if pixel == 255 else '1' for pixel in pixels])  # Convert to binary sequence
    return binary_seq

# Function to save binary sequence to a text file
def save_binary_sequence_as_text(binary_seq, file_path):
    with open(file_path, 'a') as f:  # Open in append mode ('a')
        f.write(binary_seq)  # Write the binary sequence without additional characters

# List of image paths (replace these with your actual image file paths)
image_paths = [
    "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\ch1\\key_iamge.png",
    "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\ch2\\key_iamge.png",
    "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\ch3\\key_iamge.png",
    "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\ch4\\key_iamge.png",
    "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\ch5\\key_iamge.png",
    "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\ch6\\key_iamge.png",
    "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\ch7\\key_iamge.png",
    "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\ch8\\key_iamge.png",
    "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\mri1\\key_iamge.png",
    "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\mri2\\key_iamge.png",
    "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\mri3\\key_iamge.png",
    "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\mri4\\key_iamge.png",
    "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\mri5\\key_iamge.png",
    "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\mri6\\key_iamge.png",
    "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\mri7\\key_iamge.png",
    "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\mri8\\key_iamge.png"
]

# Text file to save the binary data
text_file = "./combined_binary_key_new.txt"  # File to save the combined binary data

# Open the text file in write mode once
with open(text_file, 'w') as f:
    for image_path in image_paths:
        binary_seq_image = image_to_binary(image_path)
        # Save binary sequence to text file
        save_binary_sequence_as_text(binary_seq_image, text_file)
        print(f"Appended binary data from {image_path} to {text_file}")
        print(f"Total bits from this image: {len(binary_seq_image)}")
        print(f"Total bytes from this image: {len(binary_seq_image) // 8}")

print(f"All binary data from 8 images has been saved in text file: {text_file}")
