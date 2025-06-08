import numpy as np
from randomness import RandomnessTestSuite

# Example: Convert image to a binary sequence
def image_to_binary(image_path):
    from PIL import Image

    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)  # Convert to NumPy array
    binary_sequence = ''.join(format(pixel, '08b') for row in img_array for pixel in row)
    
    return binary_sequence

# Load your image key
image_path = "K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\fromZero3\\ch1\\key_iamge.png"
binary_data = image_to_binary(image_path)

# Convert binary string to list of integers (0,1)
binary_sequence = [int(bit) for bit in binary_data]

# Initialize the randomness test suite
rts = RandomnessTestSuite(binary_sequence)

# Perform Nonoverlapping Template Matching Test
p_value_template = rts.non_overlapping_template_match()

# Perform Binary Matrix Rank Test
p_value_rank = rts.binary_matrix_rank()

# Perform Maurer’s Universal Statistical Test
p_value_maurer = rts.maurers_universal()

# Perform Random Excursions Variant Test
p_value_excursion = rts.random_excursions_variant()

# Print results
print(f"Nonoverlapping Template Matching P-value: {p_value_template}")
print(f"Binary Matrix Rank P-value: {p_value_rank}")
print(f"Maurer's Universal Statistical P-value: {p_value_maurer}")
print(f"Random Excursions Variant P-value: {p_value_excursion}")

# Check randomness condition (P-value ≥ 0.01)
if all(p >= 0.01 for p in [p_value_template, p_value_rank, p_value_maurer, p_value_excursion]):
    print("The image key is statistically random.")
else:
    print("The image key is not random enough. Consider improving your generation process.")
