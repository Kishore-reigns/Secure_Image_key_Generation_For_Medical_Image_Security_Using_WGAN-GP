import numpy as np
from PIL import Image
height , width = 256 , 256
for i in range(10):
    noise = np.random.randint(0,256,(height,width),dtype=np.uint8)
    image = Image.fromarray(noise,mode='L')
    image.save(f"{i+1}.png")
