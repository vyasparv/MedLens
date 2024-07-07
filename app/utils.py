import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def preprocess_image(image, target_size):
    # Load the image and resize it
    image = load_img(image, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Expand dims to add batch size
    image = image / 255.0  # Normalize to [0, 1]
    return image
