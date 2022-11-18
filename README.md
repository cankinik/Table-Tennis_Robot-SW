Table Tennis Project (SW side)
This project aims to lay the software foundation of a future table tennis robot that plays against a player.
The steps for this are:
    -Construct stereo camera setup
    -Calibrate both cameras, and perform stereo calibration
    -Use undistortion maps to get rid of distortions such as fish-eye effect
    -Calculate relative pose of primary camera with a single checkerboard image laid on the ground
    -Calculate image coordinate of the ping-pong ball through each camera using HSV filtering and center of mass calculation
    -Use triangulation to find the 3D position of the ball in camera coordinates
    -Use pose and translation to convert camera coordinates into world coordinates
    -Use heuristics to detect when the ball is in the air
    -Collect samples of the ball's world coordinates in the air and form a batch
    -Fit 2nd order polynomial to this batch to estimate trajectory
The mechanical/robotics aspects needed to use this software and have a functioning robot requires:
    -Designing and assembling the robot
    -Using kinematics such that the robot can be given commands for manipulation in the 3D space
    -Using the predicted trajectory to move a robot arm such that it intersects the ball at the right time and position
    -Doing so with an attack angle and velocity that keeps the ball inside the court, and providing adequate challenge to the opponent
