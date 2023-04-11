from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf

app = Flask(__name__)

# Load the model from the file
with open('C:/Users/aman mhendiratta/OneDrive/Desktop/project/webapp/Model/model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET'])
def home():
    return render_template('Index3.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})
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

@app.route('/feedback', methods=['POST'])
def feedback():
    # Get the image and corrected label from the request
    image = request.files.get('image').read()
    corrected_label = request.form.get('corrected_label')

    # Preprocess the image
    image = tf.image.decode_image(image, channels=1)
    image = tf.image.resize(image, [28, 28])
    image = tf.cast(image, tf.float32) / 255.0

    # Update the model with the corrected label
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    corrected_label_index = labels.index(int(corrected_label))
    target = tf.one_hot(corrected_label_index, depth=len(labels))
    target = tf.reshape(target, [1, len(labels)])
    with tf.GradientTape() as tape:
        predictions = model(tf.expand_dims(image, 0))
        loss = tf.keras.losses.categorical_crossentropy(target, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Return a success message
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
