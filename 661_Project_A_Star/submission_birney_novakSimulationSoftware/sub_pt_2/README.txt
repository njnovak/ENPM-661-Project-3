Instructions:

To run this simulation, launch the world by typing

roslaunch roslaunch project3_p2 a_star.launch
and then start searching and sending commands with

python3 a_start.py

Make sure you are in the proper directories for each

We did not implement a system to alter the location or orientation of the robots start position in gazebo from python.
Therefore the start location is locked at 

start_x = 0.25
start_y = 0.25
start_theta = 0

The video provided has the configuration of

goal_x = 9
goal_y = 6
goal_theta = 0 

If you choose to do a far away goal point, I recommend commenting out the animate 
function call to speed up execution

In the submission_video.mp4, skip to 3:45 if you want to see the robot move

The accuracy of the robot in the gazebo model depends on different goal points, but overall it can maneuvre to different spots. 
Any other inquires can be sent to tbirney@umd.edu
Operation can be seen in the included gazebo_sim.mkv file
Code can be found at: https://github.com/njnovak/ENPM-661-Project-3
See each file's comments for further explanations
