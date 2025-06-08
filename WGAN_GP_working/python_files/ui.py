import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import shutil
from datetime import datetime
import torchvision.transforms as transforms

from wgp import Generator


import torch._utils

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _rebuild_tensor_v2(*args, **kwargs):
    print(f"[DEBUG] _rebuild_tensor_v2 called with {len(args)} args")
    try:
        if isinstance(args[0], torch.Tensor):
            # Already a tensor; just return it
            return args[0].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        storage = args[0]
        storage_offset = args[1]
        size = args[2]
        stride = args[3]
        requires_grad = args[4] if len(args) > 4 else False
        backward_hooks = args[5] if len(args) > 5 else None

        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        if backward_hooks:
            tensor._backward_hooks = backward_hooks
        return tensor
    except Exception as e:
        raise RuntimeError(f"Failed to rebuild tensor v2: {e}")

def _rebuild_device_tensor_from_cpu_tensor(*args, **kwargs):
    print(f"[DEBUG] _rebuild_device_tensor_from_cpu_tensor called with {len(args)} args")
    try:
        if isinstance(args[0], torch.Tensor):
            # Already a tensor; just return it
            return args[0].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Else proceed normally
        storage = args[0]
        storage_offset = args[1]
        size = args[2]
        stride = args[3]
        requires_grad = args[4] if len(args) > 4 else False
        backward_hooks = args[5] if len(args) > 5 else None
        device = args[6] if len(args) > 6 else torch.device('cpu')

        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        if backward_hooks:
            tensor._backward_hooks = backward_hooks
        return tensor.to(device)

    except Exception as e:
        raise RuntimeError(f"Failed to rebuild tensor from CPU: {e}")



torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
torch._utils._rebuild_device_tensor_from_cpu_tensor = _rebuild_device_tensor_from_cpu_tensor



# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # Converts to [0,1] range
])

to_pil = transforms.ToPILImage()

# Load model
model = torch.load('./checkpoint/fromZero3.pth', map_location=device)
generator = Generator()
generator.load_state_dict(model['generator_state_dict'])
generator.to(device)
generator.eval()



# Constants
ITERATIONS = 50

# Encryption and Decryption Functions

def arnold_cat_map_forward(image):
    print("acm forward")
    image_np = np.array(image)
    h, w, c = image_np.shape
    transformed = image_np.copy()
    for _ in range(ITERATIONS):
        new_image = np.zeros_like(transformed)
        for i in range(c):
            x, y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            new_x = (x + y) % h
            new_y = (x + 2 * y) % w
            new_image[new_x, new_y, i] = transformed[x, y, i]
        transformed = new_image.copy()
    return Image.fromarray(transformed.astype(np.uint8), mode="RGB")

def arnold_cat_map_reverse(image):
    print("acm reverse")
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
    return Image.fromarray(transformed.astype(np.uint8), mode="RGB")

def XOR(image, key):
    print("xor")
    return Image.fromarray(np.bitwise_xor(np.array(image), np.array(key)))

def encrypt(image, key):
    print("encrypt")
    scrambled = arnold_cat_map_forward(image)
    return XOR(scrambled, key)

def decrypt(image, key):
    print("decrypt")
    xored = XOR(image, key)
    return arnold_cat_map_reverse(xored)


colors = {
    "bg": "#1C1C1E",
    "frame": "#2C2C2E",
    "highlight": "#0A84FF",
    "button": "#3A3A3C",
    "text": "#FFFFFF",
    "title": "#FFD60A",
    "success": "#32CD32"
}

root = tk.Tk()
root.title("MediCrypt-ACM")
root.geometry("700x500")
root.configure(bg=colors["bg"])

def custom_button(master, text, command):
    btn = tk.Button(master, text=text, command=command,
                    bg=colors["button"], fg=colors["text"],
                    font=("Segoe UI", 12, "bold"), relief="flat",
                    activebackground=colors["highlight"],
                    padx=15, pady=10, bd=0)
    btn.bind("<Enter>", lambda e: btn.config(bg=colors["highlight"]))
    btn.bind("<Leave>", lambda e: btn.config(bg=colors["button"]))
    return btn

def set_button_success(button, text):
    button.config(text=text, bg=colors["success"])

# Frames
home_frame = tk.Frame(root, bg=colors["bg"])
encrypt_frame = tk.Frame(root, bg=colors["frame"])
decrypt_frame = tk.Frame(root, bg=colors["frame"])
for frame in (home_frame, encrypt_frame, decrypt_frame):
    frame.place(x=0, y=0, relwidth=1, relheight=1)

def show_frame(frame):
    frame.tkraise()

# Home Page
tk.Label(home_frame, text="MediCrypt-ACM", font=("Segoe UI", 24, "bold"),
         fg=colors["title"], bg=colors["bg"]).pack(pady=40)
custom_button(home_frame, "ENCRYPTION", lambda: show_frame(encrypt_frame)).pack(pady=15)
custom_button(home_frame, "DECRYPTION", lambda: show_frame(decrypt_frame)).pack(pady=15)

# === ENCRYPTION PAGE ===
tk.Label(encrypt_frame, text="🔒 ENCRYPTION", font=("Segoe UI", 18, "bold"),
         fg=colors["title"], bg=colors["frame"]).pack(pady=20)

encryption_state = {"image_path": None, "image_inserted": False, "key_inserted": False, "folder": None}
encrypt_buttons = tk.Frame(encrypt_frame, bg=colors["frame"])
encrypt_buttons.pack(pady=20)

def insert_image():
    path = filedialog.askopenfilename()
    if path:
        encryption_state["image_path"] = path
        encryption_state["image_inserted"] = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_path = os.path.join("K:/MIT_Learnings/Sem6/CIP/ui/", f"record_{timestamp}")
        os.makedirs(os.path.join(folder_path, "keys"), exist_ok=True)
        os.makedirs(os.path.join(folder_path, "encrypted"), exist_ok=True)
        encryption_state["folder"] = folder_path
        shutil.copy(path, os.path.join(folder_path, os.path.basename(path)))
        set_button_success(insert_image_btn, "Image Inserted")


def insert_key():
    path = filedialog.askopenfilename()
    if path and encryption_state["folder"]:
        encryption_state["key_inserted"] = True
        key_folder = os.path.join(encryption_state["folder"], "keys")
        shutil.copy(path, os.path.join(key_folder, os.path.basename(path)))
        set_button_success(insert_key_btn, "Key Inserted")

def generate_key():
    if not encryption_state["image_inserted"]:
        messagebox.showwarning("Warning", "Please insert an image first!")
        return
    image = Image.open(encryption_state["image_path"]).convert("RGB")
    width, height = image.size

    #random_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    tensor = generator(transform(image).unsqueeze(0)).detach().cpu().clamp(0, 1)
    tensor = tensor.squeeze(0)
    key_image = to_pil(tensor)



    key_folder = os.path.join(encryption_state["folder"], "keys")
    key_path = os.path.join(key_folder, "generated_key.png")
    key_image.save(key_path)
    encryption_state["key_inserted"] = True
    set_button_success(insert_key_btn, "Key Generated")
    messagebox.showinfo("Key Generated", f"Key saved at: {key_path}")

def encrypt_image():
    print("entered encyption")
    if not encryption_state["image_inserted"] or not encryption_state["key_inserted"]:
        messagebox.showerror("Error", "Insert image and key first!")
        return
    image = Image.open(encryption_state["image_path"]).convert("RGB")
    image = image.resize((256, 256))
    key_folder = os.path.join(encryption_state["folder"], "keys")
    key_path = os.path.join(key_folder, os.listdir(key_folder)[0])
    key = Image.open(key_path)
    print("opened key and image")
    encrypted = encrypt(image, key)
    print("encerypted")
    save_path = os.path.join(encryption_state["folder"], "encrypted", "encrypted.png")
    print(f"saved to path{save_path}")
    encrypted.save(save_path)
    encrypted.show()
    set_button_success(encrypt_btn, "Encrypted")
    messagebox.showinfo("Encrypted", f"Encrypted image saved at: {save_path}")

def reset_encrypt():
    insert_image_btn.config(text="Insert Image", bg=colors["button"])
    insert_key_btn.config(text="Insert Key", bg=colors["button"])
    encrypt_btn.config(text="Encrypt", bg=colors["button"])
    encryption_state.update({"image_path": None, "image_inserted": False, "key_inserted": False, "folder": None})

insert_image_btn = custom_button(encrypt_buttons, "Insert Image", insert_image)
insert_image_btn.grid(row=0, column=0, padx=15, pady=10)

insert_key_btn = custom_button(encrypt_buttons, "Insert Key", insert_key)
insert_key_btn.grid(row=0, column=1, padx=15, pady=10)

custom_button(encrypt_buttons, "Generate Key", generate_key).grid(row=1, column=0, padx=15, pady=10)

encrypt_btn = custom_button(encrypt_buttons, "Encrypt", encrypt_image)
encrypt_btn.grid(row=1, column=1, padx=15, pady=10)

custom_button(encrypt_frame, "Reset", reset_encrypt).pack(pady=10)
custom_button(encrypt_frame, "Back", lambda: show_frame(home_frame)).pack(pady=10)

# === DECRYPTION PAGE ===
tk.Label(decrypt_frame, text="🔓 DECRYPTION", font=("Segoe UI", 18, "bold"),
         fg=colors["title"], bg=colors["frame"]).pack(pady=20)

decryption_state = {"encrypted_path": None, "key_path": None, "image_inserted": False, "key_inserted": False, "folder": None}
decrypt_buttons = tk.Frame(decrypt_frame, bg=colors["frame"])
decrypt_buttons.pack(pady=20)

def load_encrypted():
    path = filedialog.askopenfilename()
    if path:
        decryption_state["encrypted_path"] = path
        decryption_state["image_inserted"] = True
        decryption_state["folder"] = os.path.dirname(os.path.dirname(path))
        set_button_success(load_image_btn, "Encrypted Loaded")

def insert_decrypt_key():
    path = filedialog.askopenfilename()
    if path and decryption_state["folder"]:
        decryption_state["key_path"] = path
        decryption_state["key_inserted"] = True
        key_folder = os.path.join(decryption_state["folder"], "decrypted_keys")
        os.makedirs(key_folder, exist_ok=True)
        shutil.copy(path, os.path.join(key_folder, os.path.basename(path)))
        set_button_success(insert_key_btn2, "Key Loaded")

def decrypt_image():
    if not (decryption_state["image_inserted"] and decryption_state["key_inserted"]):
        messagebox.showerror("Error", "Insert both encrypted image and key.")
        return
    image = Image.open(decryption_state["encrypted_path"]).convert("RGB")
    key_folder = os.path.join(decryption_state["folder"], "decrypted_keys")
    key_path = os.path.join(key_folder, os.listdir(key_folder)[0])
    key = Image.open(key_path).convert("RGB").resize(image.size)
    
    print(f"Image size: {image.size}, Key size: {key.size}")  # Debugging line

    # Ensure decrypted folder exists
    decrypted_folder = os.path.join(decryption_state["folder"], "decrypted")
    os.makedirs(decrypted_folder, exist_ok=True)

    decrypted = decrypt(image, key)
    save_path = os.path.join(decrypted_folder, "decrypted.png")
    decrypted.save(save_path)
    decrypted.show()
    set_button_success(decrypt_btn, "Decrypted")
    messagebox.showinfo("Decrypted", f"Decrypted image saved at: {save_path}")

def reset_decrypt():
    load_image_btn.config(text="Load Encrypted", bg=colors["button"])
    insert_key_btn2.config(text="Insert Key", bg=colors["button"])
    decrypt_btn.config(text="Decrypt", bg=colors["button"])
    decryption_state.update({"encrypted_path": None, "key_path": None, "image_inserted": False, "key_inserted": False, "folder": None})

load_image_btn = custom_button(decrypt_buttons, "Load Encrypted", load_encrypted)
load_image_btn.grid(row=0, column=0, padx=15, pady=10)

insert_key_btn2 = custom_button(decrypt_buttons, "Insert Key", insert_decrypt_key)
insert_key_btn2.grid(row=0, column=1, padx=15, pady=10)

decrypt_btn = custom_button(decrypt_buttons, "Decrypt", decrypt_image)
decrypt_btn.grid(row=1, column=0, padx=15, pady=10)

custom_button(decrypt_frame, "Reset", reset_decrypt).pack(pady=10)
custom_button(decrypt_frame, "Back", lambda: show_frame(home_frame)).pack(pady=10)

# Launch UI
show_frame(home_frame)
root.mainloop()