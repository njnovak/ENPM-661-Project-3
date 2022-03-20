#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from copy import deepcopy
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import sys


# In[ ]:


# node class that each spot in the map will occupy
# cell location and goal_location are tuples representing index 
# of current cell location and goal cell locations
class Node:
    def __init__(self, parent, cell_location, c2c, c2g=None, h=None):
        self.parent = parent
        self.cell_location = cell_location
        self.c2c = c2c
        self.c2g = c2g
        self.h = c2c+c2g


# In[ ]:


# given 2 points of a line, retrun a lambda function which caluclates the 
# y value of an x
def generate_line_eq(p1, p2):
    
    x1 = p1[0]
    y1 = p1[1]
    
    x2 = p2[0]
    y2 = p2[1]
    
    m = (y2-y1)/(x2-x1)
    b = y1-m*x1
    
    lin_func = lambda x: m*x+b
    
    return lin_func


# In[2]:


# hardcoded obstacles defined by their vertices and origins
# we just see if the current x and y are within bounding lines
def check_obstacle(x, y):
    
    # check circle
    if y <= 225 and y >= 145 and x <= np.sqrt(40**2 - (y-185)**2) + 300 and x >= -np.sqrt(40**2 - (y-185)**2) + 300:
        return True
    
    # check triangles
    t1_line = generate_line_eq((36, 185), (115, 210))
    t2_line = generate_line_eq((36, 185), (102, 100))
    t3_line = generate_line_eq((80, 180), (115, 210))
    t4_line = generate_line_eq((80, 180), (105, 100))
    if x in range(36, 81):
        if y in range(int(np.floor(t2_line(int(x)))), int(np.ceil(t1_line(int(x))))+1):
            return True
    if x in range(80, 116):
        if y in range(int(np.floor(t3_line(int(x)))), int(np.ceil(t1_line(int(x))))+1):
            return True
    if x in range(80, 106):
        if y in range(int(np.floor(t2_line(int(x)))), int(np.ceil(t4_line(int(x))))+1):
            return True
    
    # check hexagon
    x_h = 200
    y_h = 100
    a = 35
    s = int(a*2/np.sqrt(3))
    
    h1_line = generate_line_eq((x_h-a, y_h+s/2), (x_h,y_h+s))
    h2_line = generate_line_eq((x_h-a, y_h-s/2), (x_h,y_h-s))
    h3_line = generate_line_eq((x_h,y_h+s), (x_h+a, y_h+s/2))
    h4_line = generate_line_eq((x_h,y_h-s), (x_h+a, y_h-s/2))
    
    if x in range(x_h-a, x_h+1):
        if y in range(int(np.floor(h2_line(int(x)))), int(np.ceil(h1_line(int(x))))+1):
            return True
    if x in range(x_h, x_h+a+1):
            if y in range(int(np.floor(h4_line(int(x)))), int(np.ceil(h3_line(int(x))))+1):
                return True

    return False



# In[3]:


# iterate over the board, and if the cell is an obstacle, generate 
# the a circle of points around it which are padding
def generate_margin(color_map, radius):

    for y in range(len(color_map)):
        for x in range(len(color_map[y])):

            # read the color map and check to see if the current space is an obstacle
            if (color_map[y][x][0] == 255 and color_map[y][x][1] == 0 and 
                color_map[y][x][2] == 0):

                # generate circle bounds for a point if it is an obstacle
                x_range = range(x-radius, x+radius+1)

                for x_i in x_range:
                    y_upper_limit = np.ceil(np.sqrt(radius**2-(x_i-x)**2) + y)
                    y_lower_limit = np.floor(-np.sqrt(radius**2-(x_i-x)**2) + y)

                    y_range = np.arange(y_lower_limit, y_upper_limit+1)
                    for y_i in y_range:
                        if (x_i >= 0 and x_i <= 399) and (y_i >= 0 and y_i <= 249):
                            if not (color_map[int(y_i)][x_i][0] == [255] and 
                                    color_map[int(y_i)][x_i][1] == [0] and 
                                    color_map[int(y_i)][x_i][2] == [0]):
                                color_map[int(y_i)][x_i] = [0,255,0]
    return color_map


# In[4]:


# read the board and depending on each nodes status
# write the proper color in a numpy array as BGR colors
def create_color_map(height, width, radius):

    color_map = np.zeros(shape=[height, width, 3], dtype=np.uint8)

    for row in range(height):
        for col in range(width):
            if check_obstacle(col, row):
                color_map[row][col][0] = 255
                color_map[row][col][1] = 0
                color_map[row][col][2] = 0

            else:
                color_map[row][col][0] = 0
                color_map[row][col][1] = 0
                color_map[row][col][2] = 0


    color_map = generate_margin(color_map, radius)
    return color_map


# In[5]:


# will be used when iterating over closed nodes
# updates the previous color map given the current node to a specifies color
def update_color_map(curr_node, color_map,  brg_color):

    row = curr_node.cell_location[0]
    col = curr_node.cell_location[1]

    color_map[row][col][0] = brg_color[0]
    color_map[row][col][1] = brg_color[1]
    color_map[row][col][2] = brg_color[2]

    return color_map


# In[6]:


# def rounder(number, thresh):
#     first_round = round(number,1)
#     if number-np.floor(number) <= thresh


# In[7]:


# create the board
# returns a 3d array
# dimensions are height width and angle. Takes in a compressed version of 
# the height width and angle which handles the region/node similarity
def create_board(width, height, region_width = 1):
    # thresh = input("What is the node threshold? ")
    thresh = 0.5
    compressed_height = np.floor(height/thresh)
    compressed_width = np.floor(width/thresh)
    compressed_angle = np.floor(360/30)


    board = []
    for row_num in range(0, round(compressed_height)):
        temp_row = []
        for col_num in range(0, round(compressed_width)):

            temp_configuration = []
            for angle in range(round(compressed_angle)):

                c2c = np.Infinity
                c2g = np.Infinity

                new_node = Node(parent=None, 
                                c2c=c2c,
                                c2g=c2g,
                                h=c2c,
                                cell_location=[row_num, col_num, angle])

                temp_configuration.append(new_node)
            temp_row.append(temp_configuration)
        board.append(temp_row)

    return board


# In[8]:


# checking the left most command ie +60 degrees
def check_l2(curr_node, board, thresh, goal_location):
    
    curr_row = curr_node.cell_location[0]
    curr_col = curr_node.cell_location[1]
    curr_angle = curr_node.cell_location[2]
    print(curr_row,curr_col,curr_angle)

    theta_res = (curr_angle + 60)%360
    
    # TODO: Change to new round func.

    x_res = int(np.floor(curr_col + np.cos(np.deg2rad(theta_res))))
    y_res = int(np.floor(curr_row + np.sin(np.deg2rad(theta_res))))


    # check left and right border
    if x_res < 0 or x_res > 400-thresh:
        return None

    # check up and down border
    if y_res < 0 or y_res > 250-thresh:
        return None

    # if resultant space is obstacle or margin
    if ((color_map[y_res][x_res][0] == 255 and 
        color_map[y_res][x_res][1] == 0 and 
        color_map[y_res][x_res][2] == 0) or
       (color_map[y_res][x_res][0] == 0 and 
        color_map[y_res][x_res][1] == 255 and 
        color_map[y_res][x_res][2] == 0)):

        return None

    c2c=curr_node.c2c + 1
    c2g=np.sqrt((x_res-goal_location[0])**2 + (y_res-goal_location[1])**2)
    h = c2g+c2c

 
    compressed_height = int(np.floor(y_res/thresh))
    compressed_width = int(np.floor(x_res/thresh))
    compressed_angle = 0 if theta_res == 0 else int(np.floor(360/theta_res))-1

    print("Board size: ", len(board),len(board[0]),len(board[0][0]))
    print("X Y and theta : ", x_res,y_res,compressed_angle)
    new_node = board[compressed_height][compressed_width][compressed_angle]
    if h < new_node.h:
        new_node.parent = curr_node
        new_node.c2c = c2c
        new_node.c2g = c2g
        new_node.h = h
    return new_node


# In[9]:


# checking the left most command ie +60 degrees
def check_l1(curr_node, board, thresh, goal_location):
    
    curr_row = curr_node.cell_location[0]
    curr_col = curr_node.cell_location[1]
    curr_angle = curr_node.cell_location[2]
    print(curr_row,curr_col,curr_angle)

    theta_res = (curr_angle + 30)%360
    
    # TODO: Change to new round func.

    x_res = int(np.floor(curr_col + np.cos(np.deg2rad(theta_res))))
    y_res = int(np.floor(curr_row + np.sin(np.deg2rad(theta_res))))


    # check left and right border
    if x_res < 0 or x_res > 400-thresh:
        return None

    # check up and down border
    if y_res < 0 or y_res > 250-thresh:
        return None

    # if resultant space is obstacle or margin
    if ((color_map[y_res][x_res][0] == 255 and 
        color_map[y_res][x_res][1] == 0 and 
        color_map[y_res][x_res][2] == 0) or
       (color_map[y_res][x_res][0] == 0 and 
        color_map[y_res][x_res][1] == 255 and 
        color_map[y_res][x_res][2] == 0)):

        return None

    c2c=curr_node.c2c + 1
    c2g=np.sqrt((x_res-goal_location[0])**2 + (y_res-goal_location[1])**2)
    h = c2g+c2c

 
    compressed_height = int(np.floor(y_res/thresh))
    compressed_width = int(np.floor(x_res/thresh))
    compressed_angle = 0 if theta_res == 0 else int(np.floor(360/theta_res))-1

    print("Board size: ", len(board),len(board[0]),len(board[0][0]))
    print("X Y and theta : ", x_res,y_res,compressed_angle)
    new_node = board[compressed_height][compressed_width][compressed_angle]
    if h < new_node.h:
        new_node.parent = curr_node
        new_node.c2c = c2c
        new_node.c2g = c2g
        new_node.h = h
    return new_node


# In[10]:


# checking the left most command ie +60 degrees
def check_m(curr_node, board, thresh, goal_location):
    
    curr_row = curr_node.cell_location[0]
    curr_col = curr_node.cell_location[1]
    curr_angle = curr_node.cell_location[2]
    print(curr_row,curr_col,curr_angle)
    theta_res = (curr_angle)%360
    x_res = int(np.floor(curr_col + np.cos(np.deg2rad(theta_res))))
    y_res = int(np.floor(curr_row + np.sin(np.deg2rad(theta_res))))

    # check left and right border
    if x_res < 0 or x_res > 400-thresh:
        return None

    # check up and down border
    if y_res < 0 or y_res > 250-thresh:
        return None

    # if resultant space is obstacle or margin
    if ((color_map[y_res][x_res][0] == 255 and 
        color_map[y_res][x_res][1] == 0 and 
        color_map[y_res][x_res][2] == 0) or
       (color_map[y_res][x_res][0] == 0 and 
        color_map[y_res][x_res][1] == 255 and 
        color_map[y_res][x_res][2] == 0)):

        return None

    c2c=curr_node.c2c + 1
    c2g=np.sqrt((x_res-goal_location[0])**2 + (y_res-goal_location[1])**2)
    h = c2g+c2c

 
    compressed_height = int(np.floor(y_res/thresh))
    compressed_width = int(np.floor(x_res/thresh))
    compressed_angle = 0 if theta_res == 0 else int(np.floor(360/theta_res))-1

    print("Board size: ", len(board),len(board[0]),len(board[0][0]))
    print("X Y and theta : ", x_res,y_res,compressed_angle,theta_res)
    new_node = board[compressed_height][compressed_width][compressed_angle]
    if h < new_node.h:
        new_node.parent = curr_node
        new_node.c2c = c2c
        new_node.c2g = c2g
        new_node.h = h
    return new_node


# In[11]:


# checking the left most command ie +60 degrees
def check_r1(curr_node, board, thresh, goal_location):
    
    curr_row = curr_node.cell_location[0]
    curr_col = curr_node.cell_location[1]
    curr_angle = curr_node.cell_location[2]
    print(curr_row,curr_col,curr_angle)

    theta_res = (curr_angle - 30)%360
    x_res = int(np.floor(curr_col + np.cos(np.deg2rad(theta_res))))
    y_res = int(np.floor(curr_row + np.sin(np.deg2rad(theta_res))))


    # check left and right border
    if x_res < 0 or x_res > 400-thresh:
        return None

    # check up and down border
    if y_res < 0 or y_res > 250-thresh:
        return None

    # if resultant space is obstacle or margin
    if ((color_map[y_res][x_res][0] == 255 and 
        color_map[y_res][x_res][1] == 0 and 
        color_map[y_res][x_res][2] == 0) or
       (color_map[y_res][x_res][0] == 0 and 
        color_map[y_res][x_res][1] == 255 and 
        color_map[y_res][x_res][2] == 0)):

        return None

    c2c=curr_node.c2c + 1
    c2g=np.sqrt((x_res-goal_location[0])**2 + (y_res-goal_location[1])**2)
    h = c2g+c2c

 
    compressed_height = int(np.floor(y_res/thresh))
    compressed_width = int(np.floor(x_res/thresh))
    compressed_angle = 0 if theta_res == 0 else int(np.floor(360/theta_res))-1

    print("Board size: ", len(board),len(board[0]),len(board[0][0]))
    print("X Y and theta : ", x_res,y_res,compressed_angle)
    new_node = board[compressed_height][compressed_width][compressed_angle]
    if h < new_node.h:
        new_node.parent = curr_node
        new_node.c2c = c2c
        new_node.c2g = c2g
        new_node.h = h
    return new_node


# In[12]:


# checking the left most command ie +60 degrees
def check_r2(curr_node, board,thresh, goal_location):
    
    curr_row = curr_node.cell_location[0]
    curr_col = curr_node.cell_location[1]
    curr_angle = curr_node.cell_location[2]
    print(curr_row,curr_col,curr_angle)

    theta_res = (curr_angle - 60)%360
    x_res = int(np.floor(curr_col + np.cos(np.deg2rad(theta_res))))
    y_res = int(np.floor(curr_row + np.sin(np.deg2rad(theta_res))))


    # check left and right border
    if x_res < 0 or x_res > 400-thresh:
        return None

    # check up and down border
    if y_res < 0 or y_res > 250-thresh:
        return None

    # if resultant space is obstacle or margin
    if ((color_map[y_res][x_res][0] == 255 and 
        color_map[y_res][x_res][1] == 0 and 
        color_map[y_res][x_res][2] == 0) or
       (color_map[y_res][x_res][0] == 0 and 
        color_map[y_res][x_res][1] == 255 and 
        color_map[y_res][x_res][2] == 0)):

        return None

    c2c=curr_node.c2c + 1
    c2g=np.sqrt((x_res-goal_location[0])**2 + (y_res-goal_location[1])**2)
    h = c2g+c2c

 
    compressed_height = int(np.floor(y_res/thresh))
    compressed_width = int(np.floor(x_res/thresh))
    compressed_angle = 0 if theta_res == 0 else int(np.floor(360/theta_res))-1

    print("Board size: ", len(board),len(board[0]),len(board[0][0]))
    print("X Y and theta : ", x_res,y_res,compressed_angle)
    new_node = board[compressed_height][compressed_width][compressed_angle]
    if h < new_node.h:
        new_node.parent = curr_node
        new_node.c2c = c2c
        new_node.c2g = c2g
        new_node.h = h
    return new_node


# In[13]:


# generate next possible nodes for the current one
# filter out null ones which happens in a bpundary condition
def gen_next_nodes(board,curr_node):

    new_nodes = []

    new_nodes.append(check_l2(curr_node,board,0.5,goal_location))
    new_nodes.append(check_l1(curr_node,board,0.5,goal_location))
    new_nodes.append(check_m(curr_node,board,0.5,goal_location))
    new_nodes.append(check_r1(curr_node,board,0.5,goal_location))
    new_nodes.append(check_r2(curr_node,board,0.5,goal_location))
    # new_nodes.append(check_down(board, curr_node))
    # new_nodes.append(check_left(board, curr_node))
    # new_nodes.append(check_right(board, curr_node))
    # new_nodes.append(check_up_left(board, curr_node))
    # new_nodes.append(check_up_right(board, curr_node))
    # new_nodes.append(check_down_left(board, curr_node))
    # new_nodes.append(check_down_right(board, curr_node))

    return list(filter(lambda node: node is not None, new_nodes))


# In[14]:


# this is the backtracking function
# returns a list of nodes in order to find the solution
def get_solution_path(curr_node):
    solution_path= []
    
    while curr_node:
        solution_path.insert(0, curr_node)
        curr_node = curr_node.parent
        
    return solution_path


# In[15]:


# use cv2 in order to draw how the node 
# traversal looks as well as plot the shortest path

def animate(color_map, closed_nodes, solution_path, filename):
    out = cv2.VideoWriter(f'{filename}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, (400, 250))
 
    for node in closed_nodes:
        out.write(np.flipud(update_color_map(node, color_map, [255, 255, 255])))

    for node in solution_path:
        out.write(np.flipud(update_color_map(node, color_map, [0, 255, 0])))
        
    out.release()


# In[16]:


color_map = create_color_map(height=250, width=400, radius=5)
print("Starting Board")
plt.imshow(color_map, origin='lower')


# In[38]:


start_location = [0,0,0]
goal_location = [75,100,0]


if start_location[0] not in range(0, 250) or start_location[1] not in range(0, 400):
    print("Start Location Out Of Bounds")

if goal_location[0] not in range(0, 250) or goal_location[1] not in range(0, 400):
    print("Goal Location Out Of Bounds")

if color_map[start_location[0]][start_location[1]][0] == 255 or color_map[start_location[0]][start_location[1]][1] == 255:
    print('Cannot start in obstacle or obstacle margin')

if color_map[goal_location[0]][goal_location[1]][0] == 255 or color_map[goal_location[0]][goal_location[1]][1] == 255:
    print('Cannot place goal in obstacle or obstacle margin')






# In[43]:


start_node = Node(parent = None, cell_location=start_location, c2c = 0, c2g = 0, h = None)
start_node.c2c = 0

open_nodes = [start_node]
# open_nodes = []
closed_nodes = []

found = False

print('Searching')
o_board = create_board(400,250)
while len(open_nodes) > 0:
    # generate the colors of the current board and append it to the list
    # this will be a frame of an animation
    # color_maps.append(gen_color_map(board)
    open_nodes.sort(key=lambda x: x.c2c)
    curr_node = open_nodes.pop(0)
    closed_nodes.append(curr_node)


    row = curr_node.cell_location[0]
    col = curr_node.cell_location[1]
    # print(f"Searching ({row},{col})")

    if curr_node.cell_location == goal_location:
        print('Found Solution')
        found = True
        break
    else:
        next_possible_nodes = gen_next_nodes(o_board, curr_node)
        for node in next_possible_nodes:

            appendable = True

            for o_node in open_nodes:
                if o_node == node:
                    appendable = False
                    break
            if appendable:
                for c_node in closed_nodes:
                    if c_node == node:
                        appendable = False
                        break

            if appendable:
                open_nodes.append(node)

if not found:
    print('No Solution')


# In[44]:


print('Animating Search Pattern')          
                    
# back track and animate the search and solution
solution_path = get_solution_path(curr_node)
animate(color_map, closed_nodes, solution_path, filename='search')

