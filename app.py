from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the model from the file
with open('C:/Users/aman mhendiratta/OneDrive/Desktop/project/webapp/Model/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a global variable to store the image data
image_data = None
# Initialize a variable to store the uploaded image data
uploaded_image = None

@app.route('/', methods=['GET'])
def home():
    return render_template('Index4.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    global image_data, uploaded_image
    uploaded_image = request.files.get('image').read()

    # Preprocess the image
    image = tf.image.decode_image(uploaded_image, channels=1)
    image = tf.image.resize(image, [28, 28])
    image = tf.cast(image, tf.float32) / 255.0

    # Make a prediction
    predictions = model.predict(tf.expand_dims(image, 0))
    label = tf.argmax(predictions, axis=1).numpy()[0]

    # Store the image data for later use
    image_data = image.numpy()

    # Return the label as a JSON response
    return jsonify({'label': int(label)})

# Initialize the training data with empty arrays
train_images = np.array([])
train_labels = np.array([])

@app.route('/update', methods=['POST'])
def update():
    global train_images, train_labels, image_data
    # Get the corrected label from the request
    corrected_label = int(request.form['label'])

    # Add the corrected label and image to the training data
    if train_images.size == 0:
        train_images = np.expand_dims(image_data, 0)
        train_labels = np.array([corrected_label])
    else:
        train_images = np.concatenate([train_images, np.expand_dims(image_data, 0)])
        train_labels = np.concatenate([train_labels, [corrected_label]])

    # Retrain the model on the updated training data
    model.fit(train_images, train_labels, epochs=1)

    # Return a success message
    return 'Model updated successfully'

if __name__ == '__main__':
    app.run()