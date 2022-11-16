import cv2 as cv
import numpy as np
import yaml

class Camera():
    def __init__(self, camera_source, resolution_vertical, resolution_horizontal):
        # Create capture object
        feed = cv.VideoCapture(camera_source)
        self.feed = feed
        # Flags for camera attributes can be found here: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
        # Set resolution
        feed.set(3, resolution_vertical)
        feed.set(4, resolution_horizontal)
        # Disable auto white balance and auto focus
        feed.set(cv.CAP_PROP_AUTOFOCUS, 0)
        feed.set(cv.CAP_PROP_AUTO_WB, 0)
    # Releases the camera source upon destrucor call
    def __del__(self):
        self.feed.release()
    # Pass the most recent frame if it is availabe
    def get_frame(self):
        ret, frame = self.feed.read()
        if ret:
            return frame
        else:
            print('Camera frame not ready')
            pass

camera = Camera('resources/CutVideoLeft.mp4', 1920, 1080)
frame = camera.get_frame()
while frame is not None:
    cv.imshow('Test', frame)
    if cv.waitKey(10) & 0xFF == ord('q'):     # Stop Video by pressing 'q'
        break    
    frame = camera.get_frame()

print('Test over')