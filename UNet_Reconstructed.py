import numpy as np
from PIL import Image
import os

def reconstruct_image(input_folder, output_folder):
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_filename = filename.replace("_mask", "")[:-4] + '_reconstructed.png'  # Remove "_mask" and extension
            output_path = os.path.join(output_folder, output_filename)

            # Open and process tiles
            tiles = []
            for tile_filename in sorted(os.listdir(input_path), key=lambda x: int(x.split('_')[-1].split('.')[0])):
                tile_path = os.path.join(input_path, tile_filename)
                with Image.open(tile_path) as tile_img:
                    tiles.append(tile_img)

            # Combine tiles into a single numpy array
            num_tiles = len(tiles)
            num_tiles_width = int(num_tiles ** 0.5)
            num_tiles_height = (num_tiles + num_tiles_width - 1) // num_tiles_width
            tile_size = tiles[0].size[0]  # Assuming all tiles have the same size

            # Create an empty numpy array for reconstructed image
            reconstructed_image = np.zeros((num_tiles_height * tile_size, num_tiles_width * tile_size, 3), dtype=np.uint8)

            # Paste tiles onto the numpy array
            for i, tile in enumerate(tiles):
                x = i % num_tiles_width
                y = i // num_tiles_width
                x_start = x * tile_size
                y_start = y * tile_size
                reconstructed_image[y_start:y_start+tile_size, x_start:x_start+tile_size, :] = np.array(tile)

            # Convert numpy array to PIL image and save
            reconstructed_image = Image.fromarray(reconstructed_image)
            reconstructed_image = reconstructed_image.crop((0, 0, 4024, 4024))
            reconstructed_image.save(output_path)

input_folder = "Output"
output_folder = "Reconstructed"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Reconstruct images
reconstruct_image(input_folder, output_folder)


