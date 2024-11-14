# U-Net-Based-Medical-Image-Segmentation-Model

# Imports and Parameters

import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from glob import glob
from sklearn.metrics import accuracy_score

# Parameters

IMG_HEIGHT = 128

IMG_WIDTH = 128

BATCH_SIZE = 32

EPOCHS = 50

Libraries:

numpy: Used for numerical operations and array handling.

cv2: OpenCV library for image processing tasks.

os: For directory and file handling.

tensorflow: The main library for building and training machine learning models.

layers and models: Used to build the U-Net architecture.

ImageDataGenerator: For data augmentation.

train_test_split: To split data into training and validation sets.

glob: To find pathnames matching a specified pattern.

accuracy_score: To evaluate the model's performance.

Parameters: Set constants for image dimensions, batch size, and the number of training epochs.

# Synthetic Data Generation Function

def create_synthetic_data(image_dir, mask_dir, num_images=100):
    
    os.makedirs(image_dir, exist_ok=True)
    
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(num_images):
    
        # Create a synthetic grayscale image
        
        image = np.random.randint(0, 256, (IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        
        cv2.imwrite(os.path.join(image_dir, f'image_{i}.png'), image)

        # Create a corresponding binary mask (random for demonstration)
        mask = (image > 128).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(mask_dir, f'mask_{i}.png'), mask)

Function Purpose: Generates synthetic grayscale medical images and corresponding binary masks.

Directory Creation: Uses os.makedirs to create directories for images and masks if they do not exist.

Image Generation:


Creates a random grayscale image using numpy.

Generates a corresponding binary mask by thresholding the image (values greater than 128 are set to 255).

File Saving: Saves both images and masks to the specified directories.

# U-Net Model Definition

def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 1)):
    inputs = layers.Input(input_size)

    # Contracting path
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # (Similar blocks for c2, c3, c4...)

    # Bottleneck
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)

    # Expansive path
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    # (Similar blocks for u7, u8, u9...)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model
Function Purpose: Defines the U-Net architecture for image segmentation.
Architecture:
Contracting Path: Series of convolutional layers followed by max pooling to reduce spatial dimensions and capture features.
Bottleneck: The deepest part of the U-Net where the highest number of filters is used.
Expansive Path: Transpose convolutional layers (up-convolutions) followed by concatenation with corresponding contracting path layers, enabling high-resolution feature recovery.
Output Layer: A single convolutional layer with a sigmoid activation function, suitable for binary segmentation tasks.
4. Data Loading Function
python

Copy
def load_data(image_dir, mask_dir):
    images = []
    masks = []
    image_files = glob(os.path.join(image_dir, '*.png'))

    for img_path in image_files:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        images.append(img)

        mask_path = os.path.join(mask_dir, os.path.basename(img_path).replace('image', 'mask'))
        if os.path.exists(mask_path):
            mask = tf.keras.preprocessing.image.load_img(mask_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
            mask = tf.keras.preprocessing.image.img_to_array(mask) / 255.0
            masks.append(mask)
        else:
            print(f'Mask not found for {mask_path}')

    return np.array(images), np.array(masks)
Function Purpose: Loads images and masks from specified directories and preprocesses them.
Image Loading:
Uses glob to find all image files.
Each image is loaded, resized, and normalized to the range [0, 1].
Mask Loading: Corresponding masks are loaded similarly. If a mask is not found, a warning is printed.
Return Values: Returns numpy arrays of images and masks.
5. Main Execution Flow
python

Copy
# Create synthetic data
image_dir = 'synthetic_images'
mask_dir = 'synthetic_masks'
create_synthetic_data(image_dir, mask_dir)

# Load your data
images, masks = load_data(image_dir, mask_dir)

# Proceed only if images and masks are loaded
if images.size > 0 and masks.size > 0:
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Compile the model
    model = unet_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model with data augmentation
    model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
              validation_data=(X_val, y_val),
              epochs=EPOCHS,
              steps_per_epoch=len(X_train) // BATCH_SIZE)

    # Evaluate the model
    val_predictions = model.predict(X_val)
    val_predictions = (val_predictions > 0.5).astype(np.uint8)

    # Calculate accuracy (or any other metric)
    accuracy = accuracy_score(y_val.flatten(), val_predictions.flatten())
    print(f'Validation accuracy: {accuracy:.2f}')

    else:

        print("No data to process. Please check your directories.")

Synthetic Data Creation: Calls the function to generate synthetic images and masks.

Data Loading: Loads the generated images and masks into memory.

Data Check: Ensures there is data to process before proceeding.

Data Splitting: Uses train_test_split to divide the data into training and validation sets.

Data Augmentation: Sets up an ImageDataGenerator to apply real-time data augmentation during training.

Model Compilation: Compiles the U-Net model with the Adam optimizer and binary cross-entropy loss.

Model Training: Trains the model using the augmented data generator, validating on the validation set.

Model Evaluation: Predicts on the validation set and calculates accuracy, printing the result.

# Summary
This code provides a full pipeline for generating synthetic medical images, constructing a U-Net model for segmentation, training the model with augmented data, and evaluating its performance. It can be adapted for real medical image datasets by replacing the synthetic data generation with actual data loading and preprocessing.

