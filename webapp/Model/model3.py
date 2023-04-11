from flask import Flask, request, jsonify
import pickle
import tensorflow as tf

app = Flask(__name__)

# Load the model from the file
with open('C:/Users/aman mhendiratta/OneDrive/Desktop/project/webapp/Model/model.pkl', 'rb') as file:
    model = pickle.load(file)

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

if __name__ == '__main__':
    app.run(debug=True)
