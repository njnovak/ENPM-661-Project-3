r = 0.033*40
L = 0.138*40

import rospy
from geometry_msgs.msg import Twist
import math

def calc_vels(command):
    rospy.init_node('a_star_turtle')
    cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(10)

    rads_per_s = ((command[0]) + (command[1]))/2


    d_lin = (r*rads_per_s/40)

    # if (command[0]) == (command[1]):
    #     d_lin = d_lin*2


    d_theta = (r/L)*(command[1]-command[0])

    for num in range(10):
        print(f"X_d: {d_lin}, Th_d: {d_theta}")


        move_cmd = Twist()
        move_cmd.linear.x = d_lin
        move_cmd.angular.z = d_theta

        cmd_vel.publish(move_cmd)
        rate.sleep()



# set all veocities out to 0
def stop_bot():
    cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    move_cmd = Twist()
    move_cmd.linear.x = 0
    move_cmd.angular.z = 0
    cmd_vel.publish(move_cmd)


commands = []
with open('commands_3.txt', 'r') as file:
    commands = file.readlines()

actual_com = []

for com in commands:
    # com[0] = com[0].split('\n')[0]
    test = com.split(',')
    test[1] = test[1].split('\n')[0]

    test[0] = float(test[0])
    test[1] = float(test[1])

    actual_com.append(test)


print(actual_com)


for com in actual_com:
    calc_vels(com)

stop_bot()