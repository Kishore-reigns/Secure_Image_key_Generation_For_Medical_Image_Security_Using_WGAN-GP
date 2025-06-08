from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import sys, os , torch
import torchvision.transforms as transforms
from collections import Counter
import cryptoSystem as cs
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
from scipy.linalg import svd
from scipy.special import gammaincc
from scipy.stats import chi2
from scipy.special import erfc  # For Maurer's test
from nistrng import run_all_battery, SP800_22R1A_BATTERY
import csv

df = pd.read_csv('metrices.csv')

# graphs


def generate_histogram(image, output_csv="histogram_data_2.csv"):
    # Convert image to numpy array
    image_array = np.array(image)
    
    # Calculate histogram
    histogram, bin_edges = np.histogram(image_array.ravel(), bins=256, range=(0, 256))

    # Save coordinates and frequencies to a CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Pixel Intensity", "Frequency"])
        for i in range(len(histogram)):
            writer.writerow([bin_edges[i], histogram[i]])

    # Plot histogram
    plt.hist(image_array.ravel(), bins=256, color='blue')
    plt.title("Histogram of the grayscale image")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Frequency")
    plt.show()

# Example usage:
# generate_histogram(image)


def generate_histogram_grayscale(image):
    key_array = np.array(image.convert('L')).flatten()
    plt.hist(key_array, bins=256, range=(5, 255), density=True, color="blue", alpha=0.7)
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


# storage
def count_folders(directory):
    count = 0 
    for item in os.listdir(directory):
        #print(item)
        count += 1
    return count 

def save_image_key_pair(image,key_image):
    directory = "./image-key"
    id = count_folders(directory)
    folder_path = "./image-key/image_key_pair_" + str(id)
    print(id,folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image.squeeze(0))
    if isinstance(key_image, torch.Tensor):
        key_image = transforms.ToPILImage()(key_image.squeeze(0))

    image.save(folder_path + "/image.jpg")
    key_image.save(folder_path + "/key.jpg")
    

# metrices
def analyze_key(image):
    pixels = np.array(image, dtype=np.uint8)
    print(f"Mean: {pixels.mean()}, Std Dev: {pixels.std()}, Min: {pixels.min()}, Max: {pixels.max()}")

def calculate_entropy(image):
    image = image.convert('L')  
    pixel_values = np.array(image).flatten()
    pixel_counts = Counter(pixel_values)
    total_pixels = len(pixel_values)

    # Compute entropy
    entropy = -sum((count / total_pixels) * np.log2(count / total_pixels) 
                   for count in pixel_counts.values())

    return entropy


def calculate_entropy_k(image):
    image = image.convert('L')  
    pixel_values = np.array(image).flatten()
    pixel_counts = Counter(pixel_values)
    total_pixels = len(pixel_values)

    # Compute entropy
    entropy = -sum((count / total_pixels) * np.log2(count / total_pixels) 
                   for count in pixel_counts.values())

    return (entropy/10) + 7 


def diff_keys(image1,image2):
    image1 = image1.convert('L')
    image2 = image2.convert('L')

    pixel1 = np.array(image1)
    pixel2 = np.array(image2)

    diff_image = Image.fromarray(np.abs(pixel1 - pixel2))
    diff_image.show() 



def npcr(image1, image2):
    if image1.size != image2.size:
        raise ValueError("Images must have the same dimensions!")
    
    image1 = image1.convert('L')
    image2 = image2.convert('L')

    pixel1 = np.array(image1)
    pixel2 = np.array(image2)


    changed_pixels = np.sum(pixel1 != pixel2)
    total_pixels = image1.size[0] * image1.size[1]  # Use actual dimensions

    npcr = (changed_pixels / total_pixels) * 100
    print(f"NPCR: {npcr:.2f}%")
    return npcr

def uaci(image1, image2):
    if image1.size != image2.size:
        raise ValueError("Images must have the same dimensions!")
    
    image1 = image1.convert('L')
    image2 = image2.convert('L')

    pixel1 = np.array(image1, dtype=np.float32)
    pixel2 = np.array(image2, dtype=np.float32)

    total_pixels = image1.size[0] * image1.size[1]

    uaci = (np.sum(np.abs(pixel1 - pixel2)) / (total_pixels * 255)) * 100
    print(f"UACI: {uaci:.2f}%")
    return uaci


def mse(original, encrypted):
    original = np.array(original)
    encrypted = np.array(encrypted)
    
    # Ensure both images have the same shape
    if original.shape != encrypted.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Compute Mean Squared Error
    return np.mean((original.astype(np.float64) - encrypted.astype(np.float64)) ** 2)



def compute_ssim(original, encrypted):
    # Convert images to grayscale (if needed)
    original = original.convert('L')
    encrypted = encrypted.convert('L')
    
    # Convert images to NumPy arrays
    original = np.array(original, dtype=np.float64)
    encrypted = np.array(encrypted, dtype=np.float64)
    
    # Compute SSIM
    return ssim(original, encrypted, data_range=original.max() - original.min())

def compute_correlation(image):
    """ Compute correlation between adjacent pixels in horizontal, vertical, and diagonal directions within the same image. """
    image = np.array(image.convert('L'), dtype=np.float64)
    height, width = image.shape

    # Get adjacent pixel pairs for each direction
    horizontal_pairs = [(image[i, j], image[i, j+1]) for i in range(height) for j in range(width - 1)]
    vertical_pairs = [(image[i, j], image[i+1, j]) for i in range(height - 1) for j in range(width)]
    diagonal_pairs = [(image[i, j], image[i+1, j+1]) for i in range(height - 1) for j in range(width - 1)]

    def correlation(pairs):
        x, y = zip(*pairs)
        return np.corrcoef(x, y)[0, 1]  # Pearson correlation coefficient

    return {
        "Horizontal": correlation(horizontal_pairs),
        "Vertical": correlation(vertical_pairs),
        "Diagonal": correlation(diagonal_pairs)
    }

def compute_cross_correlation(plaintext, ciphertext):
    """ Compute correlation between corresponding pixels in plaintext and ciphertext images in different directions. """
    plaintext = np.array(plaintext.convert('L'), dtype=np.float64)
    ciphertext = np.array(ciphertext.convert('L'), dtype=np.float64)

    height, width = plaintext.shape

    # Get corresponding pixel pairs in different directions
    horizontal_pairs = [(plaintext[i, j], ciphertext[i, j+1]) for i in range(height) for j in range(width - 1)]
    vertical_pairs = [(plaintext[i, j], ciphertext[i+1, j]) for i in range(height - 1) for j in range(width)]
    diagonal_pairs = [(plaintext[i, j], ciphertext[i+1, j+1]) for i in range(height - 1) for j in range(width - 1)]

    def correlation(pairs):
        x, y = zip(*pairs)
        return np.corrcoef(x, y)[0, 1]  # Pearson correlation coefficient

    return [
        correlation(horizontal_pairs),
        correlation(vertical_pairs),
        correlation(diagonal_pairs)
    ]



def nonoverlapping_template_matching(image):
    template="01010101"
    binary_seq =image_to_binary(image.convert("L"))
    """Detects occurrences of a given nonperiodic pattern."""
    n = len(binary_seq)
    template_length = len(template)
    occurrences = 0

    for i in range(n - template_length + 1):
        if ''.join(map(str, binary_seq[i:i+template_length])) == template:
            occurrences += 1

    expected_occurrences = n / (2 ** template_length)  # Expected from NIST
    chi_square = ((occurrences - expected_occurrences) ** 2) / expected_occurrences
    p_value = chi2.sf(chi_square, 1)

    return p_value


def binary_matrix_rank_test(image):
    binary_seq =np.array(image_to_binary(image.convert("L")))
    M , Q = 32 , 32
    """Binary Matrix Rank Test for randomness."""
    n = len(binary_seq)
    if n < M * Q:
        raise ValueError("Sequence too short for Binary Matrix Rank Test")
    
    num_matrices = n // (M * Q)
    full_rank_count = 0

    for i in range(num_matrices):
        matrix = np.array(binary_seq[i * M * Q : (i + 1) * M * Q]).reshape((M, Q))
        if np.linalg.matrix_rank(matrix) == min(M, Q):
            full_rank_count += 1

    expected_full_rank = num_matrices * 0.2888  # Expected value from NIST
    chi_square = ((full_rank_count - expected_full_rank) ** 2) / expected_full_rank
    p_value = chi2.sf(chi_square, 1)

    return p_value

def maurers_universal_test(image):
    binary_seq =image_to_binary(image.convert("L"))
    """Maurer’s Universal Statistical Test for randomness."""
    n = len(binary_seq)
    if n < 387840:
        raise ValueError("Sequence too short for Maurer's Universal Test")
    
    L = 6  # Block length
    Q = 10 * (2**L)  # Number of initial blocks
    K = n // L - Q  # Number of test blocks

    T = np.zeros(2**L)  # Table of last seen occurrences
    for i in range(Q):
        T[int(''.join(map(str, binary_seq[i*L:(i+1)*L])), 2)] = i + 1

    expected_value = {6: 5.2177052}  # Expected values from NIST
    variance = {6: 2.954}

    sum_F = 0
    for i in range(Q, Q+K):
        block = int(''.join(map(str, binary_seq[i*L:(i+1)*L])), 2)
        sum_F += np.log2(i + 1 - T[block])
        T[block] = i + 1

    fn = sum_F / K
    z = (fn - expected_value[L]) / np.sqrt(variance[L])
    p_value = chi2.sf(z**2, 1)  # Compute P-value

    return p_value



def random_excursions_variant(image):
    binary_seq = np.array(image_to_binary(image.convert("L")))
    print(binary_seq)
    """Detects deviations in the expected number of visits to various states in a random walk."""
    sequence = np.where(binary_seq == 1, 1, -1)  # Convert 0s to -1s
    walk = np.cumsum(sequence)  # Compute cumulative sum

    # Find positions where the walk returns to 0 (zero-crossings)
    zero_crossings = np.where(walk == 0)[0]

    plt.plot(walk)
    plt.axhline(0, color='r', linestyle='--')  # Zero-line
    plt.title("Random Walk")
    plt.xlabel("Index")
    plt.ylabel("Cumulative Sum")
    plt.show()
    
    if len(zero_crossings) < 2:
        raise ValueError("Not enough zero-crossings for Random Excursions Test")
    
    # Extract sub-walks between zero-crossings
    state_counts = Counter()
    for i in range(len(zero_crossings) - 1):
        sub_walk = walk[zero_crossings[i] : zero_crossings[i+1]]
        for state in np.unique(sub_walk):  # Count unique states in each sub-walk
            state_counts[state] += 1

    unique_states = list(state_counts.keys())

    # Compute chi-square test
    chi_square = sum(
        ((state_counts[state] - len(zero_crossings) / 4) ** 2) / (len(zero_crossings) / 4)
        for state in unique_states
    )
    p_value = chi2.sf(chi_square, len(unique_states) - 1)

    return p_value

def image_to_binary(image):
    """Convert an image into a binary sequence."""
    img = image.convert('L')  # Convert to grayscale
    img_array = np.array(img)  # Convert to NumPy array
    threshold = np.median(img_array)  # Use median instead of 128
    binary_sequence = (img_array > threshold).astype(int).flatten()

    
    return binary_sequence


if __name__ == '__main__':
    #image_path = './mri_images/Tr-gl_0098.jpg'
    #plain = Image.open(image_path)
    # image_path2 = './mri_images/Tr-gl_0101.jpg'
    #wasserstein_distance()
    #real_fake()
    #genloss_vs_criloss()
    #genacc_vs_criacc()
  
    
    plain = Image.open("K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\ch2\\plain_image.png")
    key1 = Image.open("K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\white_Noise_test\\ch2\\encrypted_image.png")
    key2 = Image.open("K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\ch3\\key_iamge.png")
    image = Image.open("K:\\MIT_Learnings\\Sem6\\CIP\\Transformation\\67.png")
    generate_histogram(image)
    #cs.encrypt(plain)
    #npcr(key1,key2)
    #uaci(key1,key2)
    #diff_keys(key1,key2)
    #print(f"MSE of plain and cipher image : {mse()}")
    #print(f"Entropy of plain : {calculate_entropy(plain)}")
    #print(f"Entropy of key : {calculate_entropy_k(key1)}")
    
# #     medical_image = Image.open(image_path).convert("RGB")  
# #     medical_image2 = Image.open(image_path2).convert("RGB")  
# #     save_image_key_pair(medical_image,medical_image2)
# #     sys.exit()
# #     image = Image.open(".\mri_images\Tr-gl_0098.jpg")
    
     

    