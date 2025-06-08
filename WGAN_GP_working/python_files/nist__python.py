import numpy as np
from PIL import Image


width, height = 256, 256
white_noise = np.random.randint(0, 256, (height, width), dtype=np.uint8)


img = Image.fromarray(white_noise)
img.save("nist.png")


with open("nist.bin", "wb") as f:
    f.write(white_noise.tobytes())
