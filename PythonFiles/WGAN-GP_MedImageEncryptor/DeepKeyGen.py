import torch
import numpy

model = torch.load("WGAN_GP.pth")

medical_image = "WGAN-GP_MedImageEncryptor/MCUCXR_0001_0.png"
key = model.Genrator(medical_image)
key.show()
