from tom_code import *
from turtle import distance
import rospkg
import os
import rospy
import time
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
import math
from math import atan2


# def move_forward():
#     # move forward a distance


# def rotate():
    # Rotate a set number of degrees. Negative for left

def get_inputs(solution_path):

    for node in solution_path:
        xs = node.local_path[0]
        ys = node.local_path[1]
        # if len(xs) > 0:
            


    # points_ = []
    # # Read path points from pathfinder
    # with open(filename, "r") as points:
    #     for line in points.readlines:


x = 0.0
y= 0.0
theta = 0.0
h = []
def move_robot(msg):
    global x
    global y
    global theta
    global coordinates
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    rot_q = msg.pose.pose.orientation
    coordinates = (x,y)
    (roll,pitch,theta) = euler_from_quaternion([rot_q.x,rot_q.y,rot_q.z,rot_q.w]) # Found this function, it is rather useful

sub = rospy.Subscriber("/odom",Odometry,move_robot)

# Custom objects are create for each of the possible actions: moving straight or turning left or right

class GoForward():

    def __init__(self):
        # initiliaze
        rospy.init_node('move', anonymous=False)

       
        rospy.on_shutdown(self.shutdown)

    
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        r = rospy.Rate(5);

        move_cmd = Twist()
        move_cmd.linear.x = 0.4
        move_cmd.angular.z = 0.0

        start_time_straight = time.time()
        while not rospy.is_shutdown() and (time.time() - start_time_straight < 0.2):
            self.cmd_vel.publish(move_cmd)
            r.sleep()


    def shutdown(self):
        # stop the turtlebot
        rospy.loginfo("Stop TurtleBot")
        print(h)
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

class ClockwiseTurn():
    def __init__(self):
        # initiliaze
        rospy.init_node('move', anonymous=False)


        rospy.on_shutdown(self.shutdown)

        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        r = rospy.Rate(1);

            # Twist is a datatype for velocity
        move_cmd = Twist()
        move_cmd.linear.x = 0.0
        move_cmd.angular.z = -0.2

        start_time_turn = time.time()
        while not rospy.is_shutdown() and (time.time() - start_time_turn < 0.1):
            self.cmd_vel.publish(move_cmd)
            r.sleep()


    def shutdown(self):
        rospy.loginfo("Stop TurtleBot")
        print(h)
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

class CounterClockwiseTurn():
    def __init__(self):
        # initiliaze
        rospy.init_node('move', anonymous=False)

        
        rospy.on_shutdown(self.shutdown)

      
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        r = rospy.Rate(1)

        move_cmd = Twist()
        move_cmd.linear.x = 0.0
        move_cmd.angular.z = 0.2

        start_time_turn = time.time()
        while not rospy.is_shutdown() and (time.time() - start_time_turn < 0.1):
            self.cmd_vel.publish(move_cmd)
            r.sleep()


    def shutdown(self):
        # stop the turtlebot
        rospy.loginfo("Stop TurtleBot")
        print(h)
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)


h = []
if __name__ == '__main__':
    try:
        final_goal_x = translated_ros_backtracked_points[-1][0]
        final_goal_y = translated_ros_backtracked_points[-1][1]

        for index in range(len(translated_ros_backtracked_points)-1):
            start_x = translated_ros_backtracked_points[index][0]
            start_y = translated_ros_backtracked_points[index][1]
            goal_x = translated_ros_backtracked_points[index + 1][0]
            goal_y = translated_ros_backtracked_points[index + 1][1]
            X = goal_x - round(x,2)
            Y = goal_y - round(y,2)
            angle_to_goal = math.degrees(atan2(Y,X))
            while (x - goal_x)**2 + (y - goal_y)**2 > 0.3**2:
                
                if abs(math.degrees(theta)-abs(math.degrees(atan2(goal_y - round(y,2) ,goal_x - round(x,2)))))<5:
                    
                    print('Moving forward')
                    GoForward()
                    h.append((round(x,2),round(y,2)))
                else:

                    if math.degrees(theta)-math.degrees(atan2(goal_y - round(y,2) ,goal_x - round(x,2))) > 0:
                        print('Turning clockwise')
                        ClockwiseTurn()
                        h.append((round(x,2),round(y,2)))
                    elif math.degrees(theta)-math.degrees(atan2(goal_y - round(y,2) ,goal_x - round(x,2))) < 0:
                        print('Turning counterclockwise')
                        CounterClockwiseTurn()
                        h.append((round(x,2),round(y,2)))
                    GoForward()
                    print('Moving forward along path ')
                    h.append((round(x,2),round(y,2)))

                    if (x - final_goal_x)**2 + (y - final_goal_y)**2 <= 0.2**2: 
                        print('Goal Reached.')
                        break

    except:
        print(h)
        rospy.loginfo("The Simulation is terminated.")          

