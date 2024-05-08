from PIL import Image, ImageDraw, ImageFont
import random
import cv2
import numpy as np
from flask import Flask, render_template, request
import tempfile
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
import pytesseract

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

app = Flask(__name__)

# Preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Generate adversarial example using FGSM
def generate_adversarial_example(image, label, model, epsilon=0.01):
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor(label, dtype=tf.int32)

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        loss = tf.keras.losses.sparse_categorical_crossentropy(label_tensor, prediction)

    gradient = tape.gradient(loss, image_tensor)
    perturbation = epsilon * tf.sign(gradient)

    adversarial_image = image_tensor + perturbation
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 255)  # Adjusted clipping range for images

    return adversarial_image.numpy()

# Classify image
def classify_image(image_array):
    prediction = model.predict(image_array)
    predicted_label = np.argmax(prediction)
    confidence = prediction[0][predicted_label]

    return predicted_label, confidence

# Implementing the algorithm
def blur_image(image, is_offensive, is_text_offensive):
    if is_offensive:
        if is_text_offensive:
            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Find the contours in the image
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Loop through the contours and shuffle the text regions
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if h < 30:  # filter out small regions
                    continue
                # Shuffle the text region characters
                text = image[y:y+h, x:x+w].copy()
                text = text.reshape((h, w, 3))
                np.random.shuffle(text)
                image[y:y+h, x:x+w] = text.reshape((h, w, 3))
        else:
            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Find the contours in the image
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Loop through the contours and blur the text regions
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if h < 30:  # filter out small regions
                    continue
                # Blur the text region
                image[y:y+h, x:x+w] = cv2.GaussianBlur(image[y:y+h, x:x+w], (99, 99), 30)
    return image

def shuffle_text(image, is_text_offensive):
    if is_text_offensive:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find the contours in the image
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Loop through the contours and shuffle the text regions
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < 30:  # filter out small regions
                continue
            # Shuffle the text region characters
            text = image[y:y+h, x:x+w].copy()
            text = text.reshape((h, w, 3))
            np.random.shuffle(text)
            image[y:y+h, x:x+w] = text.reshape((h, w, 3))
    return image


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" in request.files:
            # Save uploaded image to a temporary file
            uploaded_file = request.files["image"]
            temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            uploaded_file.save(temp_image.name)

            # Preprocess and classify original image
            original_image = preprocess_image(temp_image.name)
            original_label, original_confidence = classify_image(original_image)

            # Generate adversarial example
            adversarial_image = generate_adversarial_example(original_image, original_label, model)

            # Classify adversarial image
            adversarial_label, adversarial_confidence = classify_image(adversarial_image)

            # Determine if the image is offensive
            is_offensive = original_label != adversarial_label

            # Perform OCR to detect text and determine if it's offensive
            text = pytesseract.image_to_string(Image.open(temp_image.name))
            is_text_offensive = "stripper" in text.lower()  # Example offensive text detection logic

            # Blur the original image if it's classified as offensive
            blurred_image = blur_image(cv2.imread(temp_image.name), is_offensive, is_text_offensive)

            # Shuffle the text in the image if it's offensive
            shuffled_image = shuffle_text(blurred_image, is_text_offensive)

            # Save the modified image
            output_image_path = "static/output_image.jpg"
            cv2.imwrite(output_image_path, shuffled_image)

            # Render the result page with the original and modified images
            return render_template("result.html",
                                   original_class=original_label,
                                   original_confidence=original_confidence,
                                   adversarial_class=adversarial_label,
                                   adversarial_confidence=adversarial_confidence,
                                   is_offensive=is_offensive,
                                   is_text_offensive=is_text_offensive,
                                   output_image_path=output_image_path)

    # Render the main page for image upload
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
import cv2
import numpy as np
from flask import Flask, render_template, request
import tempfile
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
import pytesseract

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

app = Flask(__name__)

# Preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Generate adversarial example using FGSM
def generate_adversarial_example(image, label, model, epsilon=0.01):
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor(label, dtype=tf.int32)

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        loss = tf.keras.losses.sparse_categorical_crossentropy(label_tensor, prediction)

    gradient = tape.gradient(loss, image_tensor)
    perturbation = epsilon * tf.sign(gradient)

    adversarial_image = image_tensor + perturbation
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 255)  # Adjusted clipping range for images

    return adversarial_image.numpy()

# Classify image
def classify_image(image_array):
    prediction = model.predict(image_array)
    predicted_label = np.argmax(prediction)
    confidence = prediction[0][predicted_label]

    return predicted_label, confidence

# Implementing the algorithm
def blur_image(image, is_offensive, is_text_offensive):
    if is_offensive:
        if is_text_offensive:
            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Find the contours in the image
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Loop through the contours and shuffle the text regions
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if h < 30:  # filter out small regions
                    continue
                # Shuffle the text region characters
                text = image[y:y+h, x:x+w].copy()
                text = text.reshape((h, w, 3))
                np.random.shuffle(text)
                image[y:y+h, x:x+w] = text.reshape((h, w, 3))
        else:
            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Find the contours in the image
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Loop through the contours and blur the text regions
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if h < 30:  # filter out small regions
                    continue
                # Blur the text region
                image[y:y+h, x:x+w] = cv2.GaussianBlur(image[y:y+h, x:x+w], (99, 99), 30)
    return image
def shuffle_text(image, is_text_offensive):
    if is_text_offensive:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find the contours in the image
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Loop through the contours and shuffle the text regions
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < 30:  # filter out small regions
                continue
            # Shuffle the text region characters
            text = image[y:y+h, x:x+w].copy()
            text = text.reshape((h, w, 3))
            np.random.shuffle(text)
            image[y:y+h, x:x+w] = text.reshape((h, w, 3))
    return image


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" in request.files:
            # Save uploaded image to a temporary file
            uploaded_file = request.files["image"]
            temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            uploaded_file.save(temp_image.name)

            # Preprocess and classify original image
            original_image = preprocess_image(temp_image.name)
            original_label, original_confidence = classify_image(original_image)

            # Generate adversarial example
            adversarial_image = generate_adversarial_example(original_image, original_label, model)

            # Classify adversarial image
            adversarial_label, adversarial_confidence = classify_image(adversarial_image)

            # Determine if the image is offensive
            is_offensive = original_label != adversarial_label

            # Determine if the text is offensive (you need to implement this logic)
            is_text_offensive = False

            # Blur the original image if it's classified as offensive
            blurred_image = blur_image(cv2.imread(temp_image.name), is_offensive, is_text_offensive)

            # Save the blurred image
            blurred_image_path = "static/blurred_image.jpg"
            cv2.imwrite(blurred_image_path, blurred_image)

            # Render the result page with the original and blurred images
            return render_template("result.html",
                                   original_class=original_label,
                                   original_confidence=original_confidence,
                                   adversarial_class=adversarial_label,
                                   adversarial_confidence=adversarial_confidence,
                                   is_offensive=is_offensive,
                                   blurred_image_path=blurred_image_path)

    # Render the main page for image upload
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
