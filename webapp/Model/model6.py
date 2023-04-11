from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the model from the file
with open('C:/Users/aman mhendiratta/OneDrive/Desktop/project/webapp/Model/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the training data
(train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

@app.route('/', methods=['GET'])
def home():
    return render_template('Index4.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    image = request.files.get('image').read()

    # Preprocess the image
    image = tf.image.decode_image(image, channels=1)
    image = tf.image.resize(image, [28, 28])
    image = tf.cast(image, tf.float32) / 255.0

    # Make a prediction
    predictions = model.predict(tf.expand_dims(image, 0))
    label = tf.argmax(predictions, axis=1).numpy()[0]

    # Return the label as a JSON response
    return jsonify({'label': int(label)})


@app.route('/update', methods=['POST'])
def update():
    # Get the corrected label and image data from the request
    corrected_label = int(request.form['label'])
    image = request.files.get('image').read()

    # Preprocess the image
    image = tf.image.decode_image(image, channels=1)
    image = tf.image.resize(image, [28, 28])
    image = tf.cast(image, tf.float32) / 255.0

    # Add the corrected label and image to the training data
    train_images = np.concatenate([train_images, tf.expand_dims(image, 0)])
    train_labels = np.concatenate([train_labels, [corrected_label]])

    # Retrain the model on the updated training data
    model.fit(train_images, train_labels, epochs=1)

    # Return a success message
    return 'Model updated successfully'


if __name__ == '__main__':
    app.run()
