import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Load the pre-trained model (adjust the path to your model)
model = tf.keras.models.load_model('capstone-model2.h5')
# model.summary()
# for layer in model.layers:
#     weights, biases = layer.get_weights()
#     print(f"Layer: {layer.name}")
#     print("Weights shape:", weights.shape)
#     print("Biases shape:", biases.shape)
#     print("Weights:", weights)
#     print("Biases:", biases)
#     print("\n")

# Path to the Downloads folder (adjust if necessary)
# image_path = os.path.expanduser("~/Downloads/testimageseptoria2.png")  # Replace with your actual image filename
# image_path = os.path.expanduser("~/Downloads/testimagemold.png")  # Replace with your actual image filename
# image_path = os.path.expanduser("~/Downloads/testimagebacteria.jpg")  # Replace with your actual image filename
# image_path = os.path.expanduser("~/Downloads/testimagebligth.jpg")  # Replace with your actual image filename
image_path = os.path.expanduser("~/Downloads/testimagespider.jpg")  # Replace with your actual image filename

# Function to preprocess the image
def preprocess_image(image_path, target_size=(120, 120)):
    try:
        # Open the image file
        image = Image.open(image_path)
        
        # Resize the image to the model's expected input size
        image = image.resize(target_size, resample=Image.Resampling.LANCZOS)
        
        # Convert to RGB (if it's not already) and normalize
        image = np.array(image.convert("RGB"))
        image_array = image / 1.0  # Normalize pixel values to [0, 1]
        
        # Add batch dimension (for Keras models, the input is expected to have shape (batch_size, height, width, channels))
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Preprocess the input image
image_array = preprocess_image(image_path)

# Check if preprocessing was successful
if image_array is not None:
    # Make the prediction using the model
    prediction = model.predict(image_array)
    
    # Get the predicted class (if classification)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Print the predicted class index
    print(f"Predicted class: {predicted_class}")
    print(f"prediction: {prediction}")
else:
    print("Failed to process the image.")
