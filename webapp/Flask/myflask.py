from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.saved_model.load('my_model')

# Create a Flask app
app = Flask(__name__)

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to (28, 28)
    image = image.resize((28, 28))
    # Convert the image to a numpy array
    image = np.array(image)
    # Scale the pixel values to [0, 1]
    image = image / 255.0
    # Add a batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# Define a route for predicting on uploaded images
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image from the request
    image = request.files['image'].read()
    # Preprocess the image
    image = preprocess_image(image)
    # Serve the prediction
    prediction = model(image)
    # Get the predicted class label
    predicted_class = tf.argmax(prediction, axis=1)
    # Return the prediction as a JSON response
    response = {'predicted_class': predicted_class.numpy()[0]}
    return jsonify(response)

# Run the app
if __name__ == '__main__':
    app.run()
