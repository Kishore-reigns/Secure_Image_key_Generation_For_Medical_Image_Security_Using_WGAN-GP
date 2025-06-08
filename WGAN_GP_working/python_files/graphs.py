import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('metrices.csv')




def generate_histogram(image):
    image_array = np.array(image)
    plt.hist(image_array.ravel(),bins=256,color='blue')
    plt.title("Histogram of the grayscale image")
    plt.xlabel("Pixel intensity")
    plt.ylabel("frequency")
    plt.show()

def generate_histogram_grayscale(image):
    key_array = np.array(image.convert('L')).flatten()
    plt.hist(key_array, bins=256, range=(0, 255), density=True, color="blue", alpha=0.7)
    plt.xlabel("Pixel Value (pi)")
    plt.ylabel("Normalized Frequency")
    plt.title("Histogram of Private Key")
    plt.show()


def genloss_vs_criloss():
    x = df['Epoch']
    y1 = df['Generator Loss']
    y2 = df['Critic Loss']
    plt.plot(x,y1,  color='green',label="Generator loss")
    plt.plot(x,y2,color="red",label="Critic loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Generator loss vs. critic loss")
    plt.legend()
    plt.show()

def genacc_vs_criacc():
    x = df['Epoch']
    y1 = df['Gen_Acc']
    y2 = df['Critic_Acc']
    plt.plot(x,y1,color='green',label='Generator accuracy')
    plt.plot(x,y2,color="red",label="Critic accuracy")
    plt.title("Generator Accuracy vs. critic Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def real_fake():
    x = df['Epoch']
    y1 = df['D_real']
    y2 = df['D_fake']

    y2_min, y2_max = y2.min(), y2.max()
    y2_norm = (2 * (y2 - y2_min) / (y2_max - y2_min)) - 1

    #y1_min , y1_max = y1.min() , y1.max()
    #y1_norm = (2 * (y1 - y1_min) / (y1_max - y1_min) ) -1 
    #plt.plot(x,y1_norm,color='green',label='Score assigned to real image')
    plt.plot(x,y2_norm,color="red",label="Score assigned to fake image")
    plt.title("Score assigned to fake image by the critic")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    #plt.legend()
    plt.show()

def wasserstein_distance():
    x = df['Epoch']
    y1 = np.abs(df['Wasserstein Distance'])  # Convert negative values to positive
    plt.plot(x,y1,color='blue')
    plt.title("Wasserstein Distance between real and fake image")
    plt.xlabel("Epochs")
    plt.ylabel("Wasserstein Distance") 
    plt.legend()
    plt.show()



