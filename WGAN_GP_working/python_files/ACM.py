from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

k = 30
a = b = c = 1 
d = 2


def display(pixel_array,i):
    plt.close("all")
    plt.figure(figsize=(5, 5))
    plt.imshow(pixel_array, cmap="gray")
    plt.title(f"Iteration {i + 1}")
    plt.axis("off")
    plt.pause(0.2)


def Scambler(image):
    image = image.resize((256, 256))
    pixel_array = np.array(image)
    for i in range(k):
        temp_array = np.zeros_like(pixel_array) 
        for x in range(256):
            for y in range(256):
                newx = (a * x + b * y) % 256
                newy = (c * x + d * y) % 256
                temp_array[newx, newy] = pixel_array[x, y]
        pixel_array = temp_array.copy()
        display(pixel_array,i)
        
    
    plt.ioff()  
    scrambled_image = Image.fromarray(pixel_array)
    return scrambled_image


def UnScrambler(image):
    image = image.resize((256, 256))
    pixel_array = np.array(image)
    for i in range(k):
        temp_array = np.zeros_like(pixel_array) 
        for x in range(256):
            for y in range(256):
                newx = (d * x - b * y) % 256
                newy = ( (-1* c * x) + a * y) % 256
                temp_array[newx, newy] = pixel_array[x, y]
        pixel_array = temp_array.copy()
        display(pixel_array,i)
        
    
    plt.ioff()  
    scrambled_image = Image.fromarray(pixel_array)
    return scrambled_image

if __name__ == '__main__':
    image_path = './mri_images/Tr-gl_0098.jpg'
    image = Image.open(image_path)
    Scambled_image = Scambler(image)
    Scambled_image.show()
    UnScrambled_image = UnScrambler(Scambled_image)
    UnScrambled_image.show()
