from picamera2 import Picamera2, Preview
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('/home/.../Downloads/CMA_Real.keras') #File Path

picam2 = Picamera2()
picam2.start_preview(Preview.QTGL)
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

def preprocess_image(image):
    if image.shape[-1] == 4:
        image = image[..., :3]
    
    image = tf.image.resize(image, (28, 28))
    image = image / 255.0
    return image

def main():
    while True:
        frame = picam2.capture_array()

        input_image = preprocess_image(frame)
        input_image = np.expand_dims(input_image, axis=0)

        predictions = model.predict(input_image)
        predicted_number = np.argmax(predictions)
        prediction_confidence = np.max(predictions)

        print(f"Predicted Number: {predicted_number}', Confidence: {prediction_confidence * 100:.2f}%")

if __name__ == '__main__':
    main()
