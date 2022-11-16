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

class StereoCamera():
    def __init__(self, camera_sources, resolution):
        if resolution == 1080:
            resolution_horizontal = 1920
            resolution_vertical = 1080
        elif resolution == 720:
            resolution_horizontal = 1280
            resolution_vertical = 720
        else:
            # Raise an exception so that the object isn't created if a bad resolution is entered
            raise ValueError('Please enter a correct resolution of 1080 or 720') 
        self.image_size = (resolution_horizontal, resolution_vertical)
        self.left_camera = Camera(camera_sources[0], resolution_horizontal, resolution_vertical)
        self.right_camera = Camera(camera_sources[1], resolution_horizontal, resolution_vertical)
        self.set_calibrated_parameters()
        self.create_undistortion_maps()
        # Also find correcting translation and rotational matrices, and correct position for them as in the project
    def __del__(self):
        del self.left_camera
        del self.right_camera
    def set_calibrated_parameters(self):
        temp_file_stream = cv.FileStorage('C:/Developer/Jupyter/ComputerVisionTrial/CalibrationResults.yml', cv.FILE_STORAGE_READ)
        self.camera_matrix_1 = temp_file_stream.getNode("cameraMatrix1").mat()
        self.distortion_matrix_1 = temp_file_stream.getNode("distCoeffs1").mat()
        self.camera_matrix_2 = temp_file_stream.getNode("cameraMatrix2").mat()
        self.distortion_matrix_2 = temp_file_stream.getNode("distCoeffs2").mat()
        self.r = temp_file_stream.getNode("R").mat()
        self.f = temp_file_stream.getNode("F").mat()
        self.e = temp_file_stream.getNode("E").mat()
        self.t = temp_file_stream.getNode("T")               # There is a difference for T, figure that out when it comes to that part
        self.rot_1 = temp_file_stream.getNode("Rot1").mat()
        self.rot_2 = temp_file_stream.getNode("Rot2").mat()
        self.p_1 = temp_file_stream.getNode("P1").mat()
        self.p_2 = temp_file_stream.getNode("P2").mat()
        self.q = temp_file_stream.getNode("Q").mat()
        temp_file_stream.release()
    def create_undistortion_maps(self):
        self.left_map_x, self.left_map_y = cv.initUndistortRectifyMap(self.camera_matrix_1, self.distortion_matrix_1, self.rot_1, self.p_1, self.image_size, cv.CV_32FC1)
        self.right_map_x, self.right_map_y = cv.initUndistortRectifyMap(self.camera_matrix_2, self.distortion_matrix_2, self.rot_2, self.p_2, self.image_size, cv.CV_32FC1)
    def get_frames(self):
        return self.left_camera.get_frame(), self.right_camera.get_frame()
    def get_undistorted_frames(self):
        temp_left_frame = self.left_camera.get_frame()
        temp_right_frame = self.right_camera.get_frame()
        return cv.remap(temp_left_frame, self.left_map_x, self.left_map_y, cv.INTER_LINEAR), cv.remap(temp_right_frame, self.right_map_x, self.right_map_y, cv.INTER_LINEAR)



def main():
    # Single camera test
    # camera = Camera('resources/CutVideoLeft.mp4', 1920, 1080)
    # frame = camera.get_frame()
    # while frame is not None:
    #     cv.imshow('Test', frame)
    #     if cv.waitKey(10) & 0xFF == ord('q'):     # Stop Video by pressing 'q'
    #         break    
    #     frame = camera.get_frame()

    # Stereo camera test
    stereo_camera = StereoCamera(['resources/CutVideoLeft.mp4', 'resources/CutVideoRight.mp4'], 1080) 
    left_frame, right_frame = stereo_camera.get_undistorted_frames()
    while left_frame is not None and right_frame is not None:
        cv.imshow('Left Feed', left_frame)
        cv.imshow('Right Feed', right_frame)
        if cv.waitKey(10) & 0xFF == ord('q'):     # Stop Video by pressing 'q'
            break    
        left_frame, right_frame = stereo_camera.get_frames()
    cv.destroyAllWindows()
    del stereo_camera

if __name__ == '__main__':
    main()