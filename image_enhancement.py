from skimage import color, exposure
import numpy as np
import matplotlib.pyplot as plt
import cv2

dark_image = cv2.imread("darkimage.jpg")


# Original image
image = dark_image.copy()

# Convert the image to grayscale
gray_image = color.rgb2gray(image)

# Compute the histogram and cumulative distribution function
# (CDF) of the gray image
hist, bin_centers = exposure.histogram(gray_image)
cdf = np.cumsum(hist) / np.sum(hist)

# Split the color image into separate RGB channels
red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]

# Compute color multipliers from CDF for each channel
multipliers = 0.5 + 0.5 * np.concatenate([cdf]*3)
red_mult = multipliers[0]
green_mult = multipliers[1]
blue_mult = multipliers[2]

# Apply global histogram equalization to each channel
red_eq = exposure.equalize_hist(red_channel)
green_eq = exposure.equalize_hist(green_channel)
blue_eq = exposure.equalize_hist(blue_channel)

# Adjust channels by corresponding multipliers
red_final = red_eq * red_mult
green_final = green_eq * green_mult
blue_final = blue_eq * blue_mult

# Recombine the channels into a color image
adjusted_image = np.stack((red_final, green_final, blue_final), axis=-1)

# Create a 4x2 subplot for original and adjusted channels and images
fig, axs = plt.subplots(4, 2, figsize=(10, 20))

channels = [
    ('Red', red_channel, red_eq), 
    ('Green', green_channel, green_eq),
    ('Blue', blue_channel, blue_eq)
]

for i, (color_name, original, adjusted) in enumerate(channels):
    axs[i, 0].set_title(f'Original {color_name} Channel')
    axs[i, 0].imshow(original, cmap=f'{color_name}s')
    axs[i, 1].set_title(f'Adjusted {color_name} Channel')
    axs[i, 1].imshow(adjusted, cmap=f'{color_name}s')

# Plot the original and final combined images
axs[3, 0].set_title('Original Image')
axs[3, 0].imshow(image)
axs[3, 1].set_title('Adjusted Image')
axs[3, 1].imshow(adjusted_image)

# Show the subplots
plt.show()
