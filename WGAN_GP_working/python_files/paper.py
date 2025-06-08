
import torch
from wgp import Generator # change this
from PIL import Image
import torchvision.transforms as transforms
import time, os, Utils, cryptoSystem as cs
import numpy as np
import csv
import random
import Utils_

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


#paths = ["ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8"]
paths = ["mri1","mri2","mri3","mri4","mri5","mri6","mri7","mri8"]

# with open(metrics_path , mode="w", newline="") as file :
#     writer = csv.writer(file)
#     writer.writerow(["MSE","SSIM","Cross-Correlation-Hori","Cross-Correlation-Verti","Cross-Correlation-Diag","Plain Correlation-Hori","Plain Correlation-Verti","Plain Correlation-Diag","Cipher Correlation-Hori","Cipher Correlation-Verti","Cipher Correlation-Diag"])


for i in range(len(paths)):

    plain_image_path = "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\test_images\\" + paths[i] + ".jpg"
    print(f"image{i} : {paths[i]}")

    medical_image = Image.open(plain_image_path).convert("RGB")
    medical_image_tensor = transform(medical_image).unsqueeze(0).to(device) 


    start_time = time.time()
    key_tensor = generator(medical_image_tensor).detach().cpu().clamp(0, 1)
    end_time = time.time()


    medical_image_pil = to_pil(medical_image_tensor.squeeze(0).cpu())
    key_image_pil = to_pil(key_tensor.squeeze(0))
    #key_image_pil = Utils_.key_metrics(key_image_pil)
    # key_image_pil = cs.arnold_cat_map_forward__(key_image_pil,random.randint(1,192))
    # key_image_pil=add_white_noise(key_image_pil,200)

    scrambled_image = cs.arnold_cat_map_forward(medical_image_pil)

    encrypted_image = cs.encrypt(medical_image_pil, key_image_pil)

    reverse_xor = cs.XOR(encrypted_image,key_image_pil)


    decrypted_image = cs.decrypt(encrypted_image, key_image_pil)


    dest_path = "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\" + paths[i]
    os.makedirs(dest_path, exist_ok=True)
    
    medical_image_pil.save(os.path.join(dest_path,"plain_image.png"))
    key_image_pil.save(os.path.join(dest_path,"key_iamge.png"))
    scrambled_image.save(os.path.join(dest_path,"scrambled_image.png"))
    encrypted_image.save(os.path.join(dest_path,"encrypted_image.png"))
    reverse_xor.save(os.path.join(dest_path,"reverse_xor.png"))
    decrypted_image.save(os.path.join(dest_path,"decrypted_image.png"))


    mse = Utils.mse(medical_image_pil, encrypted_image)
    ssim = Utils.compute_ssim(medical_image_pil, encrypted_image)
    cross_corr = Utils.compute_cross_correlation(medical_image_pil, encrypted_image)
    plain_corr = Utils.compute_correlation(medical_image_pil)  
    cipher_corr = Utils.compute_correlation(encrypted_image)  


    metrics_path = "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\metrices.csv"
    with open(metrics_path , mode="a", newline="") as file :
        writer = csv.writer(file)
        writer.writerow([
        mse, ssim, 
        cross_corr[0], cross_corr[1], cross_corr[2],
        plain_corr.get("Horizontal", ""), plain_corr.get("Vertical", ""), plain_corr.get("Diagonal", ""),
        cipher_corr.get("Horizontal", ""), cipher_corr.get("Vertical", ""), cipher_corr.get("Diagonal", "")
    ])


    



  
