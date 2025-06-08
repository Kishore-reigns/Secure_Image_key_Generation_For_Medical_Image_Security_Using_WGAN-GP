import os
from PIL import Image
import matplotlib.pyplot as plt
import cryptoSystem as cs

# Base folder
root_folder = "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero4"
# "K:\\MIT_Learnings\\Sem6\\CIP\\Transformation\\67.png"

# Keys to try for each encrypted image
key_folders = {
    "Key 1": "ch1",
    "Key 6": "ch6",
    "Key 4": "ch4",
    "Key 3": "ch3"
}

# Encrypted image folders (rows) - 1st image from ch1, 2nd from ch6, etc.
encrypted_sources = {
    "1": "ch1",
    "6": "ch6",
    "4": "ch4",
    "3": "ch3"
}

# Create 4x4 plot
fig, axes = plt.subplots(4, 4, figsize=(12, 12))


# Loop through encrypted images (rows)
for row_index, (enc_label, enc_folder) in enumerate(encrypted_sources.items()):
    encrypted_path = os.path.join(root_folder, enc_folder, "encrypted_image.png")
    
    # Loop through different keys (columns)
    for col_index, (key_label, key_folder) in enumerate(key_folders.items()):
        key_path = os.path.join(root_folder, key_folder, "key_iamge.png")
        
        ax = axes[row_index, col_index]
        
        try:
            enc_img = Image.open(encrypted_path)
            key_img = Image.open(key_path)
            decrypted_img = cs.decrypt(enc_img, key_img)
            ax.imshow(decrypted_img)
        except Exception as e:
            print(f"Error: Row {enc_label} with Key {key_label} – {e}")
            ax.set_facecolor("black")
        
        ax.axis("off")
        if col_index == 0:
            ax.set_ylabel(enc_label, fontsize=12)
        if row_index == 0:
            ax.set_title(key_label, fontsize=10)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
