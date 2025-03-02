from PIL import Image
import numpy as np 

image1 = Image.open("1.png")
image2 = Image.open("medical_image.jpg").resize((253,256))

pixel1 = np.array(image1)
pixel2 = np.array(image2)

encrypted_array = np.bitwise_xor(pixel1,pixel2)

encrypted_image = Image.fromarray(encrypted_array)

encrypted_image.show()
