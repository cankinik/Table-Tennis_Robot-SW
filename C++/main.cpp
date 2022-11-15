#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector> 
#include <algorithm>
#include <sstream>


using namespace std;
using namespace cv;

// Function Declerations
void initializeCameras(VideoCapture *leftVideoFeed, VideoCapture *rightVideoFeed);
Point2f hsvCenterFinder(Mat inputFrame);
void loadCalibrationResults();
void createUndistortionMapping();
void calculateRotationAndTranslation();
vector<float> correctResultsForRotation(vector<float> resultsToBeCorrected);

// Constants and Global Variables
const int leftVideoFeedIndex = 1;
const int rightVideoFeedIndex = 0;
const int videoHorizontalResolution = 1920;
const int videoVerticalResolution = 1080;
const Size imageSize = Size(1920, 1080); 
const double squareSize = 0.038;
const Scalar lowerHSVBound = Scalar(8, 160, 160);        //Ping pong ball tresholds suitable for daytime natural lighting
const Scalar upperHSVBound = Scalar(22, 255, 255);
char userInputKey;
//Variables for calibration
Mat cameraMatrix1, distCoeffs1, R1, T1;
Mat cameraMatrix2, distCoeffs2, R2, T2;
Mat R, F, E;
Vec3d T;
Mat Rot1, Rot2, P1, P2, Q;
Mat leftMapX, leftMapY, rightMapX, rightMapY;
Mat correctingRotationalMatrix, correctingTranslationalVector;

// Main Program
int main ()
{
    VideoCapture leftVideoFeed, rightVideoFeed;
    initializeCameras(&leftVideoFeed, &rightVideoFeed);	
    Mat leftFrame, rightFrame;
    Mat temp3DPositions;
    Point2f leftCOM, rightCOM;
    float x, y, z;
    vector<Point2f> tempLeftCOMVector, tempRightCOMVector;
    vector<float> instantenousCoordinates;
    vector<vector<float>> last30Coordinates;
    loadCalibrationResults();
    createUndistortionMapping();
	calculateRotationAndTranslation();
    namedWindow("Left Feed", 0);
	namedWindow("Right Feed", 0);
    while (true)
    {
        leftVideoFeed >> leftFrame;                     //Also add the undistortion part here
        rightVideoFeed >> rightFrame;        
        leftCOM = hsvCenterFinder(leftFrame);
        rightCOM = hsvCenterFinder(rightFrame);      
        drawMarker(leftFrame, leftCOM, Scalar(0, 0, 255), cv::MARKER_CROSS, 50, 5);
        drawMarker(rightFrame, rightCOM, Scalar(0, 0, 255), cv::MARKER_CROSS, 50, 5);
        imshow("Left Feed", leftFrame);
        imshow("Right Feed", rightFrame);
		resizeWindow("Left Feed", 1280, 720);
        resizeWindow("Right Feed", 1280, 720);
        userInputKey = waitKey(1);
		if ( userInputKey == 'q' )
		{
			break;
		}	
        tempLeftCOMVector.push_back(leftCOM);
        tempRightCOMVector.push_back(rightCOM);
        triangulatePoints( P1, P2, tempLeftCOMVector, tempRightCOMVector, temp3DPositions );
        tempLeftCOMVector.clear();
        tempRightCOMVector.clear();
        x = (temp3DPositions.at<float>(0, 0))/(temp3DPositions.at<float>(3, 0));
        y = (temp3DPositions.at<float>(1, 0))/(temp3DPositions.at<float>(3, 0));
        z = (temp3DPositions.at<float>(2, 0))/(temp3DPositions.at<float>(3, 0)); 
        cout << "X: " << x << " Y: " << y << " Z: " << z << endl;
        instantenousCoordinates.clear();
        instantenousCoordinates.push_back(x * 100);
        instantenousCoordinates.push_back(y * 100);
        instantenousCoordinates.push_back(z * 100);
        if(last30Coordinates.size() == 30)
        {
            last30Coordinates.erase(last30Coordinates.begin() + 0);
        }
        last30Coordinates.push_back(instantenousCoordinates);
    }   

}

// Function Definitions
void initializeCameras(VideoCapture *leftVideoFeed, VideoCapture *rightVideoFeed)
{
	// *leftVideoFeed = VideoCapture(leftVideoFeedIndex);  //Creating instances with the given indices
	// *rightVideoFeed = VideoCapture(rightVideoFeedIndex);
    *leftVideoFeed = VideoCapture("/home/cankinik/Desktop/TableTennisRobot/Sample Video/left.avi");
    *rightVideoFeed = VideoCapture("/home/cankinik/Desktop/TableTennisRobot/Sample Video/right.avi");
	(*leftVideoFeed).set(CAP_PROP_FOURCC, 0x47504A4D);	//Using MJPG format rather than YUVY so that 30FPS 1080p is enabled
	(*rightVideoFeed).set(CAP_PROP_FOURCC, 0x47504A4D);
	(*leftVideoFeed).set(CAP_PROP_AUTOFOCUS, 0);		//Disabling autofocus so that the cameras will always see the same way
	(*rightVideoFeed).set(CAP_PROP_AUTOFOCUS, 0);
    (*leftVideoFeed).set(CAP_PROP_AUTO_WB, 0);          //Disabling autofocus so that HSV values will be constant
    (*rightVideoFeed).set(CAP_PROP_AUTO_WB, 0); 
	(*leftVideoFeed).set(CAP_PROP_FRAME_WIDTH, videoHorizontalResolution);  //Setting resolution values
	(*leftVideoFeed).set(CAP_PROP_FRAME_HEIGHT, videoVerticalResolution);
	(*rightVideoFeed).set(CAP_PROP_FRAME_WIDTH, videoHorizontalResolution);
	(*rightVideoFeed).set(CAP_PROP_FRAME_HEIGHT, videoVerticalResolution);
}

Point2f hsvCenterFinder(Mat inputFrame)
{
    Mat tempHSVFrame, tempThresholdFrame;
    cvtColor(inputFrame, tempHSVFrame, COLOR_BGR2HSV);      //Converting to HSV gamut
    blur(tempHSVFrame, tempHSVFrame, cv::Size(1, 1));       //Adding Gaussian Noise to Ssmooth out the image
    inRange(tempHSVFrame, lowerHSVBound, upperHSVBound, tempThresholdFrame);  //Obtaining the masked image using the HSV bounds and the converted image
    // imshow("Thresholded Image", tempThresholdFrame);
    Moments m = moments(tempThresholdFrame, false);                //Getting the conter of mass of the object from the masked image
    Point2f com(m.m10 / m.m00, m.m01 / m.m00);    
    return com;
}

void loadCalibrationResults()
{
	FileStorage file("CalibrationResults.yml", FileStorage::READ);	

    file["cameraMatrix1"] >> cameraMatrix1;
    file["distCoeffs1"] >> distCoeffs1;
    file["cameraMatrix2"] >> cameraMatrix2;
    file["distCoeffs1"] >> distCoeffs2;
    file["R"] >> R;
    file["F"] >> F;
    file["E"] >> E;
    file["T"] >> T;
    file["Rot1"] >> Rot1;
    file["Rot2"] >> Rot2;
    file["P1"] >> P1;
    file["P2"] >> P2;
    file["Q"] >> Q;

    file.release();
}

void createUndistortionMapping()
{
    initUndistortRectifyMap(cameraMatrix1, distCoeffs1, Rot1, P1, imageSize, CV_32FC1, leftMapX, leftMapY);
    initUndistortRectifyMap(cameraMatrix2, distCoeffs2, Rot2, P2, imageSize, CV_32FC1, rightMapX, rightMapY);
}


void calculateRotationAndTranslation()
{
	Mat checkerboardImage = imread("RotationCalibrationPicture.png");
	int CHECKERBOARD[2]{6, 9}; 
    vector<Point3f> objectPoints;
	vector<Point2f> cornerPoints;
	TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);
	Mat grayImage;
	Mat correctingRotationVector;
    
    for(int i = 0; i < CHECKERBOARD[1]; i++)
    {
        for(int j = 0; j < CHECKERBOARD[0]; j++) 
        {
            objectPoints.push_back(Point3f(j,i,0) * squareSize);
        }
    }
    
    cvtColor(checkerboardImage, grayImage, COLOR_BGR2GRAY);
    if( findChessboardCorners(grayImage, Size(CHECKERBOARD[0],CHECKERBOARD[1]), cornerPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE) )
    {
    	cornerSubPix(grayImage, cornerPoints, Size(11,11), Size(-1,-1), criteria);
    	solvePnP(objectPoints, cornerPoints, cameraMatrix1, distCoeffs1, correctingRotationVector, correctingTranslationalVector, false);
    	Rodrigues(correctingRotationVector, correctingRotationalMatrix); 
    	//These steps are switching the columns, they will probably not be necessary when we calibrate properly in terms of horizontal or landscape of the checkerboard being compliant with the dimensions
    	double temp;
    	temp = correctingRotationalMatrix.at<double>(0, 0);
    	correctingRotationalMatrix.at<double>(0, 0) = correctingRotationalMatrix.at<double>(0, 1);
    	correctingRotationalMatrix.at<double>(0, 1) = temp;
    	temp = correctingRotationalMatrix.at<double>(1, 0);
    	correctingRotationalMatrix.at<double>(1, 0) = correctingRotationalMatrix.at<double>(1, 1);
    	correctingRotationalMatrix.at<double>(1, 1) = temp;
    	temp = correctingRotationalMatrix.at<double>(2, 0);
    	correctingRotationalMatrix.at<double>(2, 0) = correctingRotationalMatrix.at<double>(2, 1);
    	correctingRotationalMatrix.at<double>(2, 1) = temp;
	}
    else
    {
    	cout << "Unable to find the checkerboard in the image, please use another image to calibrate the rotation and translation matrices." << endl;
    }
}

vector<float> correctResultsForRotation(vector<float> resultsToBeCorrected)
{
	float x, y, z, row1, row2, row3, shiftingX, shiftingY, shiftingZ;
	vector<float> result = resultsToBeCorrected;
	x = resultsToBeCorrected[0] - correctingTranslationalVector.at<double>(0, 0);
    y = resultsToBeCorrected[1] - correctingTranslationalVector.at<double>(0, 1);
    z = resultsToBeCorrected[2] - correctingTranslationalVector.at<double>(0, 2);
    //Subtracted translation vector at this point, now we should do cross product with the transpose of the rotational matrix
    row1 = x * correctingRotationalMatrix.at<double>(0, 0) + y * correctingRotationalMatrix.at<double>(1, 0) + z * correctingRotationalMatrix.at<double>(2, 0);
    row2 = x * correctingRotationalMatrix.at<double>(0, 1) + y * correctingRotationalMatrix.at<double>(1, 1) + z * correctingRotationalMatrix.at<double>(2, 1);
    row3 = x * correctingRotationalMatrix.at<double>(0, 2) + y * correctingRotationalMatrix.at<double>(1, 2) + z * correctingRotationalMatrix.at<double>(2, 2);
    //Have ( rotation_matrix.T @ ( point - translation_matrix ) ) at this point, now we need to add (rotation_matrix @ translation_matrix) * np.array([[1],[1],[0]])
    shiftingX = correctingTranslationalVector.at<double>(0, 0) * correctingRotationalMatrix.at<double>(0, 0) + correctingTranslationalVector.at<double>(0, 1) * correctingRotationalMatrix.at<double>(0, 1) + correctingTranslationalVector.at<double>(0, 2) * correctingRotationalMatrix.at<double>(0, 2);
    shiftingY = correctingTranslationalVector.at<double>(0, 0) * correctingRotationalMatrix.at<double>(1, 0) + correctingTranslationalVector.at<double>(0, 1) * correctingRotationalMatrix.at<double>(1, 1) + correctingTranslationalVector.at<double>(0, 2) * correctingRotationalMatrix.at<double>(1, 2);
    shiftingZ = correctingTranslationalVector.at<double>(0, 0) * correctingRotationalMatrix.at<double>(2, 0) + correctingTranslationalVector.at<double>(0, 1) * correctingRotationalMatrix.at<double>(2, 1) + correctingTranslationalVector.at<double>(0, 2) * correctingRotationalMatrix.at<double>(2, 2);
    x = row1 + shiftingX;
    y = row2 + shiftingY;
    z = row3;	//We will be displaying the Z coordinate with respect to the floor
    result[0] = x * 100;
    result[1] = y * 100;
    result[2] = z * -100;
	return result;
}