import torch
from wgp import Generator
from PIL import Image
import torchvision.transforms as transforms

import time, os, Utils,Utils_, cryptoSystem as cs
import numpy as np
import torch._utils
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

checkpoint = torch.load('./checkpoint/fromZero4.pth', map_location=device)

#print(checkpoint.keys())  

generator = Generator().to(device)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator = generator.to(device)
generator.eval()

# Load image
plain_image_path = './mri_images/MCUCXR_0173_1.png'
medical_image = Image.open(plain_image_path).convert("RGB")
medical_image_tensor = transform(medical_image).unsqueeze(0).to(device)

# Generate key
start_time = time.time()
key_tensor = generator(medical_image_tensor).detach().cpu().clamp(0, 1)
end_time = time.time()

# Convert tensors to images
medical_image_pil = to_pil(medical_image_tensor.squeeze(0).cpu())
medical_image_pil.show()
#key_image_pil = Utils_.key_metrics(to_pil(key_tensor.squeeze(0)))
key_image_pil = to_pil(key_tensor.squeeze(0))
key_image_pil.show()


# Encrypt and decrypt
encrypted_image = cs.encrypt(medical_image_pil, key_image_pil)
encrypted_image.show()
decrypted_image = cs.decrypt(encrypted_image, key_image_pil)
decrypted_image.show()



# Print execution time and metrics
os.system('cls' if os.name == 'nt' else 'clear')
print(f"Time to generate and transform key: { (end_time - start_time):.4f} sec")
print(f"MSE : {Utils.mse(medical_image_pil, encrypted_image)}")
print(f"SSIM : {Utils.compute_ssim(medical_image_pil, encrypted_image)}")
print(f"Cross-Correlation : {Utils.compute_cross_correlation(medical_image_pil, encrypted_image)}")
print(f"Pixel Correlation in plain image : {Utils.compute_correlation(medical_image_pil)}")
print(f"Pixel Correlation in cipher image : {Utils.compute_correlation(encrypted_image)}")
