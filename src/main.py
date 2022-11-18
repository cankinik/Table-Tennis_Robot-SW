import cv2 as cv
import numpy as np
import yaml
import glob

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
            raise ValueError('Camera not accessible') 
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
    def calculate_pose(self, image_source='resources/Rotation_Calibration_Image.png', grid_size=(6, 9), square_size=0.054):
        # Initialize variables
        horizontal, vertical = grid_size
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        object_points = np.zeros((horizontal*vertical, 3), np.float32)
        object_points[:,:2] = np.mgrid[0:horizontal,0:vertical].T.reshape(-1,2) * square_size
        # Get image and convert it to gray scale
        image = cv.imread(image_source)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (horizontal,vertical), None)
        # If found, use object and image points to find the correcting rotation and translation        
        if ret:
            ret, rvecs, tvecs = cv.solvePnP(object_points, corners, self.camera_matrix, self.distortion_matrix)        
            correcting_rotation_matrix, _ = cv.Rodrigues(rvecs)
            correcting_translation_vector = tvecs.T
            return correcting_rotation_matrix, correcting_translation_vector.T
        else:
            raise ValueError('Could not find the checkerboard to calculate correcting rotation and translation values') 
    def calculate_calibration_parameters(self, calibration_image_glob_path='resources/Calibration_Images/*.png', square_size=0.054):
        # TODO: Code from another project. Change the few bits so that it is consistent with the rest of the project. Also, this is for single camera, make the version for stereo
        # Calculate the intrinsic parameters of the camera
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Horizontal and vertical are flipped, but lazy. The numbers must be exact, and they are the corners inside, not outside
        horizontal = 7 
        vertical = 4
        objp = np.zeros((horizontal*vertical,3), np.float32)
        objp[:,:2] = np.mgrid[0:horizontal,0:vertical].T.reshape(-1,2) * square_size

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob(calibration_image_glob_path)
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (horizontal,vertical), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv.drawChessboardCorners(img, (horizontal,vertical), corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(200)
        cv.destroyAllWindows()
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, mtx, (1280,720), cv.CV_32FC1)


class Stereo_Camera():
    def __init__(self, camera_sources, resolution):
        self.set_resolution(resolution)
        # Initialize the two camera objects
        self.left_camera = Camera(camera_sources[0], self.image_size[0], self.image_size[1])
        self.right_camera = Camera(camera_sources[1], self.image_size[0], self.image_size[1])        
        self.set_calibration_parameters()    # Read camera parameters from the YML file        
        self.create_undistortion_maps()     # Create undistortion maps that can be reused
        # Correcting rotation and translation will be with respect to the left camera, as we use that as the primary camera
        self.correcting_rotation_matrix, self.correcting_translation_vector = self.left_camera.calculate_pose()
    def __del__(self):
        del self.left_camera
        del self.right_camera
    def set_resolution(self, resolution):
        # Initialize resolution-based variables
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

    def set_calibration_parameters(self):
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
        self.left_camera.camera_matrix = self.camera_matrix_1
        self.left_camera.distortion_matrix = self.distortion_matrix_1
        self.right_camera.camera_matrix = self.camera_matrix_2
        self.right_camera.distortion_matrix = self.distortion_matrix_2
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
        return np.array([[float(position[0])], [float(position[1])], [float(position[2])]])
    def correct_position_for_rotation(self, raw_position):
        result = raw_position - self.correcting_translation_vector        
        result = self.correcting_rotation_matrix.T @ result
        return result


def main():
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
        ball_corrected_position = stereo_camera.correct_position_for_rotation(ball_raw_positon)
        print(ball_corrected_position)
    cv.destroyAllWindows()
    del stereo_camera

if __name__ == '__main__':
    main()