import cv2
import depthai as dai
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('/home/.../Downloads/CMA_Real.keras') #File Path

pipeline = dai.Pipeline()
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutVideo.setStreamName("Video")

camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setVideoSize(1920,1080)
xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)
camRgb.video.link(xoutVideo.input)

#with dai.Device(pipeline) as device:
	#video = device.getOutputQueue(name="Video", maxSize=4, blocking=True)

def preprocess_image(image):
    if image.shape[-1] == 4:
        image = image[..., :3]
    
    image = tf.image.resize(image, (28, 28))
    image = image / 255.0
    return image

def main():
    with dai.Device(pipeline) as device:
        video = device.getOutputQueue(name="Video", maxSize=1, blocking=False)
        while True:
            frame =  video.get()
            videoIn = frame.getCvFrame()
        #frame.getCvFrame()
        # cv2.imshow("video", videoIn.getCvFrame())
            cv2.imshow("video", frame.getCvFrame())
            input_image = preprocess_image(videoIn)
            input_image = np.expand_dims(input_image, axis=0)

            predictions = model.predict(input_image)
            predicted_number = np.argmax(predictions)
            prediction_confidence = np.max(predictions)

            print(f"Predicted Number: {predicted_number}', Confidence: {prediction_confidence * 100:.2f}%")
            if cv2.waitKey(1) == ord('q'):
                break

if __name__ == '__main__':
    main()
