import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

def crop_and_resize_image(input_path, output_path, crop_size=(4024, 4024), final_size=(512, 512)):
    # Check whether the output folder exists and create if not
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Scroll through all images in the input folder
    for image_name in os.listdir(input_path):
        image_path = os.path.join(input_path, image_name)
        image = cv2.imread(image_path)

        # Ensure that the image has been loaded
        if image is None:
            print(f"Das Bild {image_name} konnte nicht geladen werden.")
            continue

        # Crop image
        height, width = image.shape[:2]
        if width > height:  # wider image
            start = (width - height) // 2
            cropped_image = image[:, start:start+height]
        else:  # Taller or square image
            start = (height - width) // 2
            cropped_image = image[start:start+width, :]

        # Reduce image to final size
        resized_image = cv2.resize(cropped_image, final_size, interpolation=cv2.INTER_AREA)

        # Save cropped and reduced image
        output_image_path = os.path.join(output_path, image_name)
        cv2.imwrite(output_image_path, resized_image)

def apply_unet_to_image(model, image_path):
    image = Image.open(image_path)
    image = np.array(image).astype('float32')
    image = np.expand_dims(image, axis=0)  # Extension of the dimension for the model

    # Prediction of the mask
    mask = model.predict(image)
    return mask.squeeze(0)  #removingr Batch-Dimension


def save_mask(mask, save_path):
    # Remove the batch dimension, if available
    # Make sure that the mask is in the correct format
    if len(mask.shape) == 3 and mask.shape[2] == 1:  # for 3D-Array with one channel
        mask = mask.squeeze(2)  # remove channel

    # make sure, mask has two dimension
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # convert mask into correct format
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image.save(save_path)

def process_images(input_folder, output_folder, model):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.tif'):
            image_path = os.path.join(input_folder, filename)
            mask = apply_unet_to_image(model, image_path)

            # Path for saving the mask
            save_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '_mask.png')
            save_mask(mask, save_path)
            
# Set path to the folders for savingn
input_folder = 'E:/t'
temp_folder= 'E:/'
output_folder = 'E:/'

#Crop Images
crop_and_resize_image(input_folder,temp_folder)

# Load new trained model
model = load_model("E:/Masterarbeit/UNET/Model/Current/Model1.h5",compile=False)



process_images(temp_folder,output_folder, model)
