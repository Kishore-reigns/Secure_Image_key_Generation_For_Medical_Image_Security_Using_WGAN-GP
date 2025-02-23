from PIL import Image
import numpy as np

image1 = Image.open("1.png").convert('L')
image2 = Image.open("5.png").convert('L')
width1 , height1 = image1.size
width2 , height2 = image2.size
#print(f"width :{width}px\n height :{height}px")
npcr = 0 
image1 = image1.resize((256,256))
image2 = image2.resize((256,256))

pixel1 = np.array(image1, dtype=np.float32)
pixel2 = np.array(image2, dtype=np.float32)
#print(len(pixel1) , len(pixel2))
changed_pixels = np.sum(np.abs(pixel1 - pixel2))

total_pixels = 256*256

#print(pixel1[0,0])
#print(pixel2[0,0])
uaci = (changed_pixels/(total_pixels*255))*100

print(F"The Unified Average Changin Intensiy (UACI) of the two images are  : {uaci}%" )

