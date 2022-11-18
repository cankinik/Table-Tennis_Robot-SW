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
    def get_com(self, input_frame):
        # Convert from RGB to HSV, and blur to get rid of outliers
        hsv_frame = cv.cvtColor(input_frame, cv.COLOR_BGR2HSV)
        blurred_hsv_frame = cv.blur(hsv_frame, (1, 1))
        # The HSV threshold values are hard-coded right now. Might want to change that later on
        hsv_lower_treshold_for_pingpong_ball = np.array([8, 160, 160])
        hsv_upper_treshold_for_pingpong_ball = np.array([22, 255, 255])
        # Mask anything that is not inside HSV range to only leave the ball
        masked_frame = cv.inRange(blurred_hsv_frame, hsv_lower_treshold_for_pingpong_ball, hsv_upper_treshold_for_pingpong_ball)
        # Calculate the center of mass of the left out pixels (everything except ball should be masked out) to find center of ball. 
        moment = cv.moments(masked_frame, bool(0))
        x = int(moment["m10"] / moment["m00"])
        y = int(moment["m01"] / moment["m00"])
        # Return x, y (image coordinates of the center of the ball)
        return x, y


class Stereo_Camera():
    def __init__(self, camera_sources, resolution):
        # Initialize resolution
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
        # Initialize the two camera objects
        self.left_camera = Camera(camera_sources[0], resolution_horizontal, resolution_vertical)
        self.right_camera = Camera(camera_sources[1], resolution_horizontal, resolution_vertical)        
        self.set_calibrated_parameters()    # Read camera parameters from the YML file        
        self.create_undistortion_maps()     # Create undistortion maps that can be reused
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
        # Return COM of ball from hsv masking and moment com finding
    def get_coms(self, left_frame, right_frame):
        return self.left_camera.get_com(left_frame), self.right_camera.get_com(right_frame)
    def get_raw_position(self, left_com, right_com):
        position = cv.triangulatePoints( self.p_1, self.p_2, left_com, right_com)
        position /= position[3]
        return [float(position[0]), float(position[1]), float(position[2])]


def main():
    # Single camera test
    # camera = Camera('resources/CutVideoLeft.mp4', 1920, 1080)
    # frame = camera.get_frame()
    # while frame is not None:
    #     cv.imshow('Test', frame)
    #     if cv.waitKey(10) & 0xFF == ord('q'):     # Stop Video by pressing 'q'
    #         break    
    #     frame = camera.get_frame()

    # # Stereo camera test
    # stereo_camera = Stereo_Camera(['resources/CutVideoLeft.mp4', 'resources/CutVideoRight.mp4'], 1080) 
    # left_frame, right_frame = stereo_camera.get_undistorted_frames()
    # while left_frame is not None and right_frame is not None:
    #     cv.imshow('Left Feed', left_frame)
    #     cv.imshow('Right Feed', right_frame)
    #     if cv.waitKey(10) & 0xFF == ord('q'):     # Stop Video by pressing 'q'
    #         break    
    #     left_frame, right_frame = stereo_camera.get_frames()
    # cv.destroyAllWindows()
    # del stereo_camera

    # Ball tracking test
    stereo_camera = Stereo_Camera(['resources/CutVideoLeft.mp4', 'resources/CutVideoRight.mp4'], 1080) 
    left_frame, right_frame = stereo_camera.get_undistorted_frames()
    while left_frame is not None and right_frame is not None:
        left_com, right_com = stereo_camera.get_coms(left_frame, right_frame)
        marked_left_frame = cv.drawMarker(left_frame, left_com, (0, 0, 255), cv.MARKER_CROSS, 50, 5)
        marked_right_frame = cv.drawMarker(right_frame, right_com, (0, 0, 255), cv.MARKER_CROSS, 50, 5)
        cv.imshow('Left Feed', marked_left_frame)
        cv.imshow('Right Feed', marked_right_frame)
        if cv.waitKey(10) & 0xFF == ord('q'):     # Stop Video by pressing 'q'
            break    
        left_frame, right_frame = stereo_camera.get_frames()
        ball_raw_positon = stereo_camera.get_raw_position(left_com, right_com)
        print(ball_raw_positon)
    cv.destroyAllWindows()
    del stereo_camera

if __name__ == '__main__':
    main()