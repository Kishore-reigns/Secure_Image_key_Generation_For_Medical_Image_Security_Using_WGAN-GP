import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

# Hill Cipher functions
def create_key_matrix(key_string, size=2):
    key_matrix = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            key_matrix[i][j] = ord(key_string[i * size + j]) % 26
    return key_matrix

def encrypt_hill(plain_text, key_matrix):
    size = key_matrix.shape[0]
    plain_text = plain_text.replace(" ", "").upper()
    if len(plain_text) % size != 0:
        plain_text += "X" * (size - len(plain_text) % size)  # Padding with 'X'
    cipher_text = ""
    for i in range(0, len(plain_text), size):
        block = plain_text[i:i + size]
        block = [ord(c) - ord('A') for c in block]
        block = np.dot(key_matrix, block) % 26
        cipher_text += ''.join([chr(b + ord('A')) for b in block])
    return cipher_text

def decrypt_hill(cipher_text, key_matrix):
    size = key_matrix.shape[0]
    det = int(np.linalg.det(key_matrix)) % 26
    if det == 0:
        raise ValueError("Key matrix is singular and cannot be inverted.")
    inv_key_matrix = np.linalg.inv(key_matrix) * det  # Find inverse matrix using modular inverse
    inv_key_matrix = np.round(inv_key_matrix).astype(int) % 26

    cipher_text = cipher_text.upper()
    plain_text = ""
    for i in range(0, len(cipher_text), size):
        block = cipher_text[i:i + size]
        block = [ord(c) - ord('A') for c in block]
        block = np.dot(inv_key_matrix, block) % 26
        plain_text += ''.join([chr(b + ord('A')) for b in block])
    return plain_text

# GUI Functions
def open_encryption_page():
    encryption_window = tk.Toplevel(root)
    encryption_window.title("Encryption Page")
    encryption_window.geometry("500x300")
    encryption_window.configure(bg="#34495E")
    
    ttk.Label(encryption_window, text="Encryption Page", font=("Arial", 16, "bold"), background="#34495E", foreground="white").pack(pady=20)
    
    button_frame = tk.Frame(encryption_window, bg="#34495E")
    button_frame.pack(pady=20)

    image_inserted = False
    key_inserted = False
    key_image_path = None

    # Placeholder function to simulate WGAN-GP encryption
    def wgan_gp_encrypt(image_path):
        # Simulating the WGAN-GP encryption here, you would replace it with actual WGAN-GP logic
        print(f"Simulating WGAN-GP encryption on: {image_path}")
        return image_path  # Return the same image for now as a placeholder

    def insert_image():
        nonlocal image_inserted
        image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if image_path:
            print(f"Image selected: {image_path}")
            image = Image.open(image_path).resize((120, 80))
            img_display = ImageTk.PhotoImage(image)
            insert_image_button.config(image=img_display, text="", compound="center")
            insert_image_button.image = img_display
            image_inserted = True  

    def insert_key():
        nonlocal key_inserted, key_image_path
        key_image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if key_image_path:
            print(f"Key image selected: {key_image_path}")
            key_image = Image.open(key_image_path).resize((120, 80))
            img_display = ImageTk.PhotoImage(key_image)
            insert_key_button.config(image=img_display, text="", compound="center")
            insert_key_button.image = img_display
            key_inserted = True

    def generate():
        if not image_inserted or not key_inserted:
            messagebox.showerror("Missing Files", "Please insert both image and key.")
        else:
            # Simulate encryption with WGAN-GP
            encrypted_image = wgan_gp_encrypt(key_image_path)  # Here we use key_image_path as a placeholder for the encryption
            print(f"Generated encrypted image: {encrypted_image}")

            # Apply Hill Cipher encryption (example, you can implement the full logic here)
            key_string = "GYBNQKURP"  # Hill cipher key string
            key_matrix = create_key_matrix(key_string, 3)  # Example: 3x3 key matrix
            encrypted_hill = encrypt_hill(encrypted_image, key_matrix)
            print(f"Encrypted image after Hill Cipher: {encrypted_hill}")

    insert_image_button = ttk.Button(button_frame, text="Insert Image", command=insert_image, style="TButton")
    insert_image_button.grid(row=0, column=0, padx=20, pady=10)
    
    insert_key_button = ttk.Button(button_frame, text="Insert Key", command=insert_key, style="TButton")
    insert_key_button.grid(row=0, column=1, padx=20, pady=10)

    generate_button = ttk.Button(encryption_window, text="Generate", command=generate, style="TButton")
    generate_button.pack(pady=20)
    
    ttk.Button(encryption_window, text="Back", command=encryption_window.destroy, style="TButton").pack(pady=20)

def open_decryption_page():
    decryption_window = tk.Toplevel(root)
    decryption_window.title("Decryption Page")
    decryption_window.geometry("500x300")
    decryption_window.configure(bg="#1F618D")

    ttk.Label(decryption_window, text="Decryption Page", font=("Arial", 16, "bold"), background="#1F618D", foreground="white").pack(pady=20)

    button_frame = tk.Frame(decryption_window, bg="#1F618D")
    button_frame.pack(pady=20)

    encrypted_image_inserted = False

    def load_encrypted_image():
        nonlocal encrypted_image_inserted
        image_path = filedialog.askopenfilename(filetypes=[("Encrypted Images", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if image_path:
            print(f"Encrypted Image selected: {image_path}")
            image = Image.open(image_path).resize((120, 80))
            img_display = ImageTk.PhotoImage(image)
            load_image_button.config(image=img_display, text="", compound="center")
            load_image_button.image = img_display
            encrypted_image_inserted = True  

    def decrypt():
        if not encrypted_image_inserted:
            messagebox.showerror("Missing File", "Please load an encrypted image first.")
        else:
            # Apply Hill Cipher decryption first (example, you can implement the full logic here)
            key_string = "GYBNQKURP"  # Hill cipher key string (same key used for encryption)
            key_matrix = create_key_matrix(key_string, 3)  # Example: 3x3 key matrix
            decrypted_hill = decrypt_hill("encrypted_image_example", key_matrix)
            print(f"Decrypted image after Hill Cipher: {decrypted_hill}")

            # Simulate decryption with WGAN-GP (you would reverse the WGAN-GP process)
            print("Simulating WGAN-GP decryption...")

    def reset():
        nonlocal encrypted_image_inserted
        load_image_button.config(image='', text="Load Encrypted Image")
        encrypted_image_inserted = False

    load_image_button = ttk.Button(button_frame, text="Load Encrypted Image", command=load_encrypted_image, style="TButton")
    load_image_button.grid(row=0, column=0, padx=20, pady=10)
    
    decrypt_button = ttk.Button(button_frame, text="Decrypt", command=decrypt, style="TButton")
    decrypt_button.grid(row=0, column=1, padx=20, pady=10)

    reset_button = ttk.Button(decryption_window, text="Reset", command=reset, style="TButton")
    reset_button.pack(pady=10)

    exit_button = ttk.Button(decryption_window, text="Exit", command=decryption_window.destroy, style="TButton")
    exit_button.pack(pady=10)

root = tk.Tk()
root.title("BATMAN")
root.geometry("500x300")
root.resizable(False, False)
root.configure(bg="#2C3E50")

title_label = ttk.Label(root, text="BATMAN", font=("Arial", 16, "bold"), background="#2C3E50", foreground="white")
title_label.pack(pady=20)

button_frame = tk.Frame(root, bg="#2C3E50")
button_frame.pack(pady=30)

style = ttk.Style()
style.configure("TButton", font=("Arial", 12, "bold"), padding=10)
style.map("TButton", background=[("active", "#2980B9")], foreground=[("active", "white")])

encrypt_button = ttk.Button(button_frame, text="ENCRYPTION", command=open_encryption_page, style="TButton")
encrypt_button.grid(row=0, column=0, padx=20, pady=10)

decrypt_button = ttk.Button(button_frame, text="DECRYPTION", command=open_decryption_page, style="TButton")
decrypt_button.grid(row=0, column=1, padx=20, pady=10)

root.mainloop()
