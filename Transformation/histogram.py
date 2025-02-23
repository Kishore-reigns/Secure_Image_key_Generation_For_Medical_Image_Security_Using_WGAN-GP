from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np

image = Image.open("1.png")
image_array = np.array(image)

#print(image_array)
#print(image_array.ndim)

plt.hist(image_array.ravel(),bins=256,color='blue')
plt.title("Histogram of the grayscale image")
plt.xlabel("Pixel intensity")
plt.ylabel("frequency")
plt.show()


