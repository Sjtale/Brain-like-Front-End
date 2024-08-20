import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image):
    if image is not None:
        # Define the types of noise
        noise_types =  ['defocus','gaussian','poisson','salt_and_pepper','glass', 'motion','zoom']
        # noise_type = np.random.choice(noise_types)
        noise_type='defocus'
        noisy_image = image.copy()
        noisy_image = np.array(image)
        if noise_type == 'gaussian':
            # Add Gaussian noise
            mean = 0
            std = 25
            gaussian_noise = np.random.normal(mean, std, noisy_image.shape).astype(np.uint8)
            noisy_image = cv2.add(noisy_image, gaussian_noise)

        elif noise_type == 'poisson':
            # Add Poisson noise
            noisy_image = np.random.poisson(noisy_image)

        elif noise_type == 'salt_and_pepper':
            # Add salt and pepper noise
            noise_density = 0.000002
            width, height = image.size  # Get width and height using the size attribute
            num_pixels = int(width * height * noise_density)
            salt_coordinates = [np.random.randint(0, width - 1, num_pixels),
                                np.random.randint(0, height - 1, num_pixels)]
            pepper_coordinates = [np.random.randint(0, width - 1, num_pixels),
                                  np.random.randint(0, height - 1, num_pixels)]
            noisy_image[salt_coordinates] = 255
            noisy_image[pepper_coordinates] = 0

        elif noise_type == 'defocus':
            # Add defocus blur
            kernel_size = 15
            blurry_image = cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), 0)
            noisy_image = blurry_image

        elif noise_type == 'glass':
            # Simulate glass blur
            scale = 10
            height, width = noisy_image.shape[:2]
            x, y = np.mgrid[0:height, 0:width]
            x_distort = x + scale * np.sin(2 * np.pi * x / height)
            y_distort = y + scale * np.sin(2 * np.pi * y / width)
            x_distort = x_distort.astype(np.float32)  # Convert to float32
            y_distort = y_distort.astype(np.float32)  # Convert to float32
            distorted_indices = cv2.remap(noisy_image, x_distort, y_distort, interpolation=cv2.INTER_LINEAR)
            noisy_image = distorted_indices

        elif noise_type == 'motion':
            # Add motion blur
            kernel_size = 15
            angle = -45
            motion_blur_kernel = np.zeros((kernel_size, kernel_size))
            motion_blur_kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            motion_blur_kernel = cv2.warpAffine(
                motion_blur_kernel, cv2.getRotationMatrix2D((kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle, 1.0),
                (kernel_size, kernel_size))
            motion_blur_kernel = motion_blur_kernel / kernel_size

            # Convert the image to the correct data type
            noisy_image = noisy_image.astype(np.float32)

            # Apply motion blur
            noisy_image = cv2.filter2D(noisy_image, -1, motion_blur_kernel)

            # Convert back to uint8 after filtering
            noisy_image = noisy_image.astype(np.uint8)


        elif noise_type == 'zoom':
            # Simulate zoom blur
            scale = 0.8
            height, width = noisy_image.shape[:2]
            center = (width / 2, height / 2)
            zoom_matrix = cv2.getRotationMatrix2D(center, 0, scale)
            noisy_image = cv2.warpAffine(noisy_image, zoom_matrix, (width, height), flags=cv2.INTER_LINEAR)
        
        noisy_image = Image.fromarray(noisy_image)
        # Save the noisy image
        # Convert NumPy array to PIL Image
        
        return noisy_image
    else:
        print("Image loading failed. Please provide a valid image file path.")

    



import os

# Define paths
input_folder = 'val'
output_folder = 'val_noise'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through all subdirectories and files in the input folder
for root, dirs, files in os.walk(input_folder):
    for file in files:
        # Create the corresponding directory in the output folder
        output_dir = os.path.join(output_folder, os.path.relpath(root, input_folder))
        os.makedirs(output_dir, exist_ok=True)

        # Read the image using PIL
        img_path = os.path.join(root, file)
        original_image = Image.open(img_path)

        # Apply noise
        noisy_image = add_salt_and_pepper_noise(original_image)

        # Save the noisy image to the output folder
        output_path = os.path.join(output_dir, file)
        noisy_image.save(output_path)

print("Noise addition and saving completed.")