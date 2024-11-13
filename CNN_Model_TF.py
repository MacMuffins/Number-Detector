import numpy as np
import os, cv2, itertools
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D

# Load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Convert grayscale images to RGB by duplicating the grayscale data across the three channels
train_images_rgb = np.stack([train_images]*3, axis=-1) / 255.0
test_images_rgb = np.stack([test_images]*3, axis=-1) / 255.0
# These images are inherintly 28 x 28

train_images_inverted = 1 - train_images_rgb
test_images_inverted = 1 - test_images_rgb

# Stack the original and inverted images along the first axis (batch axis)
train_images_rgb = np.concatenate((train_images_rgb, train_images_inverted), axis=0)
test_images_rgb = np.concatenate((test_images_rgb, test_images_inverted), axis=0)

# Also, concatenate labels, as both original and inverted images have the same labels
train_labels = np.concatenate((train_labels, train_labels))
test_labels = np.concatenate((test_labels, test_labels))

model = Sequential()
model.add(Conv2D(32, (7,7), input_shape=(28, 28, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3,3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(train_images_rgb, train_labels, epochs=5, batch_size=500, validation_data = (test_images_rgb, test_labels))

print("Evaluate on train data")
Yhat = model.predict(train_images_rgb)
predicted_classes = np.argmax(Yhat, axis=1)
acc = np.mean(predicted_classes == train_labels)
print("The train accuracy rate is:", acc * 100)

print("Evaluate on test data")
Yhat_test = model.predict(test_images_rgb)
predicted_classes = np.argmax(Yhat_test, axis=1)
acc = np.mean(predicted_classes == test_labels)
print("The test accuracy rate is:", acc * 100)

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
    
def predict_and_display(file_path):
    test_image = read_image(file_path)
    X_img = test_image.reshape(-1, 28, 28, 3) / 255
    
    prediction = model.predict(X_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    prediction_confidence = np.max(prediction)
    
    print(f"Predicted Class: {predicted_class}, Confidence: {prediction_confidence * 100:.2f}%")

    plt.imshow(cv2.cvtColor(test_image.astype('uint8'), cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted Class: {predicted_class}")
    plt.show()

def main():
    model.save('CMA_Real.keras')
    return

if __name__ == '__main__':
    main()
