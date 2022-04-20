from tom_code import *
from turtle import distance
import rospkg
import os
import rospy
from nav_msgs.msg import Odometry
import math


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
class GoForward():

    def __init__(self):
        # initiliaze
        rospy.init_node('move', anonymous=False)

        # tell user how to stop TurtleBot
        # rospy.loginfo("To stop TurtleBot CTRL + C")

            # What function to call when you ctrl + c
        rospy.on_shutdown(self.shutdown)

        # Create a publisher which can "talk" to TurtleBot and tell it to move
            # Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        #TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
        r = rospy.Rate(5);

            # Twist is a datatype for velocity
        move_cmd = Twist()
        # let's go forward at 0.2 m/s
        move_cmd.linear.x = 0.4
        # let's turn at 0 radians/s
        move_cmd.angular.z = 0.0

        start_time_straight = time.time()
        # as long as you haven't ctrl + c keeping doing...
        while not rospy.is_shutdown() and (time.time() - start_time_straight < 0.2):
            # publish the velocity
            self.cmd_vel.publish(move_cmd)
        # wait for 0.1 seconds (10 HZ) and publish again
            r.sleep()


    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop TurtleBot")
        print(h)
    # a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
        self.cmd_vel.publish(Twist())
    # sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
        rospy.sleep(1)

class ClockwiseTurn():
    def __init__(self):
        # initiliaze
        rospy.init_node('move', anonymous=False)

        # tell user how to stop TurtleBot
        # rospy.loginfo("To stop TurtleBot CTRL + C")

            # What function to call when you ctrl + c
        rospy.on_shutdown(self.shutdown)

        # Create a publisher which can "talk" to TurtleBot and tell it to move
            # Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        #TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
        r = rospy.Rate(1);

            # Twist is a datatype for velocity
        move_cmd = Twist()
        # let's go forward at 0.2 m/s
        move_cmd.linear.x = 0.0
        # let's turn at 0 radians/s
        move_cmd.angular.z = -0.2

        start_time_turn = time.time()
        # as long as you haven't ctrl + c keeping doing...
        while not rospy.is_shutdown() and (time.time() - start_time_turn < 0.1):
            # publish the velocity
            self.cmd_vel.publish(move_cmd)
        # wait for 0.1 seconds (10 HZ) and publish again
            r.sleep()


    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop TurtleBot")
        print(h)
    # a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
        self.cmd_vel.publish(Twist())
    # sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
        rospy.sleep(1)

class CounterClockwiseTurn():
    def __init__(self):
        # initiliaze
        rospy.init_node('move', anonymous=False)

        # tell user how to stop TurtleBot
        # rospy.loginfo("To stop TurtleBot CTRL + C")

            # What function to call when you ctrl + c
        rospy.on_shutdown(self.shutdown)

        # Create a publisher which can "talk" to TurtleBot and tell it to move
            # Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        #TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
        r = rospy.Rate(1);

            # Twist is a datatype for velocity
        move_cmd = Twist()
        # let's go forward at 0.2 m/s
        move_cmd.linear.x = 0.0
        # let's turn at 0 radians/s
        move_cmd.angular.z = 0.2

        start_time_turn = time.time()
        # as long as you haven't ctrl + c keeping doing...
        while not rospy.is_shutdown() and (time.time() - start_time_turn < 0.1):
            # publish the velocity
            self.cmd_vel.publish(move_cmd)
        # wait for 0.1 seconds (10 HZ) and publish again
            r.sleep()


    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop TurtleBot")
        print(h)
    # a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
        self.cmd_vel.publish(Twist())
    # sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
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
                    # print('goal now is ', goal_x,',',goal_y)
                    # print(abs(math.degrees(atan2(goal_y - round(y,2) ,goal_x - round(x,2)))- math.degrees(theta)))
                    print('Moving forward')
                    GoForward()
                        # print('angle to the goal ',goal_x,goal_y, 'is : ',angle_to_goal)
                    h.append((round(x,2),round(y,2)))
                else:

                        # print('going to rotate : matching ',math.degrees(theta),'to',angle_to_goal)
                    if math.degrees(theta)-math.degrees(atan2(goal_y - round(y,2) ,goal_x - round(x,2))) > 0:
                        # print 'angle is > : ', math.degrees(theta)-abs(math.degrees(atan2(goal_y - round(y,2) ,goal_x - round(x,2))))
                        print('Turning clockwise')
                        ClockwiseTurn()
                        # print 'reducing angle to ' , abs(math.degrees(theta)-abs(math.degrees(atan2(goal_y - round(y,2) ,goal_x - round(x,2)))))
                        h.append((round(x,2),round(y,2)))
                    elif math.degrees(theta)-math.degrees(atan2(goal_y - round(y,2) ,goal_x - round(x,2))) < 0:
                        # print 'angle is > : ', math.degrees(theta)-abs(math.degrees(atan2(goal_y - round(y,2) ,goal_x - round(x,2))))
                        print('Turning counterclockwise')
                        CounterClockwiseTurn()
                        # print 'reducing angle to ' , abs(math.degrees(theta)-abs(math.degrees(atan2(goal_y - round(y,2) ,goal_x - round(x,2)))))
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


