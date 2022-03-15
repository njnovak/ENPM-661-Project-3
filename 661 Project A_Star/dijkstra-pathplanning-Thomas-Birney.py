import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from copy import deepcopy
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import sys


# node class that each spot in the map will occupy
# cell location and goal_location are tuples representing index 
# of current cell location and goal cell locations
class Node:
    def __init__(self, parent, c2c, is_obstacle, is_margin, cell_location):
        self.parent = parent
        self.c2c = c2c
        self.is_obstacle = is_obstacle
        self.is_margin = is_margin
        self.cell_location = cell_location


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


# iterate over the board, and if the cell is an obstacle, generate 
# the a circle of points around it which are padding
def generate_margin(board, radius):

    for y in range(len(board)):
        for x in range(len(board[y])):

            if board[y][x].is_obstacle:

                # generate circle bounds for a point if it is an obstacle
                x_range = range(x-radius, x+radius+1)

                for x_i in x_range:
                    y_upper_limit = np.ceil(np.sqrt(radius**2-(x_i-x)**2) + y)
                    y_lower_limit = np.floor(-np.sqrt(radius**2-(x_i-x)**2) + y)

                    y_range = np.arange(y_lower_limit, y_upper_limit+1)
                    for y_i in y_range:
                        if (x_i >= 0 and x_i <= 399) and (y_i >= 0 and y_i <= 249):
                            board[int(y_i)][x_i].is_margin = True


# create the board
# returns a 2d array

def create_board(width, height, margin):

    board = []
    for row_num in range(0, height):
        temp_row = []
        for col_num in range(0, width):

            c2c = np.Infinity
            is_obstacle = check_obstacle(col_num, row_num)

            new_node = Node(parent=None, 
                            c2c=c2c, 
                            is_obstacle=is_obstacle,
                            is_margin=False,
                            cell_location=[row_num, col_num])
    
            temp_row.append(new_node)
        board.append(temp_row)
    generate_margin(board, margin)

    return board


# read the board and depending on each nodes status
# write the proper color in a numpy array as BGR colors
def create_color_map(board):

    color_map = np.zeros(shape=[250, 400, 3], dtype=np.uint8)

    for row in range(250):
        for col in range(400):
            
            if board[row][col].is_margin and not board[row][col].is_obstacle:
                color_map[row][col][0] = 0
                color_map[row][col][1] = 0
                color_map[row][col][2] = 255

            elif board[row][col].is_obstacle:
                color_map[row][col][0] = 255
                color_map[row][col][1] = 0
                color_map[row][col][2] = 0

            else:
                color_map[row][col][0] = 0
                color_map[row][col][1] = 0
                color_map[row][col][2] = 0
    
    return color_map


# will be used when iterating over closed nodes
# updates the previous color map given the current node to a specifies color
def update_color_map(curr_node, color_map,  brg_color):

    row = curr_node.cell_location[0]
    col = curr_node.cell_location[1]

    color_map[row][col][0] = brg_color[0]
    color_map[row][col][1] = brg_color[1]
    color_map[row][col][2] = brg_color[2]

    return color_map


def check_up(board, curr_node):
    
    row = curr_node.cell_location[0]
    col = curr_node.cell_location[1]
    
    # check out of bounds
    if row < 249:
        
        # check if obstacle
        new_node = board[row+1][col]
        
        if not new_node.is_margin:
            
            new_c2c = 1 + curr_node.c2c
            
            if new_c2c < new_node.c2c:
                new_node.c2c = new_c2c
                new_node.parent = curr_node
                
            return new_node

    return None


def check_down(board, curr_node):
    
    row = curr_node.cell_location[0]
    col = curr_node.cell_location[1]
    
    # check out of bounds
    if row > 0:
        
        # check if obstacle
        new_node = board[row-1][col]
        
        if not new_node.is_margin:
            
            new_c2c = 1 + curr_node.c2c
            
            if new_c2c < new_node.c2c:
                new_node.c2c = new_c2c
                new_node.parent = curr_node
                
            return new_node

    return None


def check_left(board, curr_node):
    
    row = curr_node.cell_location[0]
    col = curr_node.cell_location[1]
    
    # check out of bounds
    if col > 0:
        
        # check if obstacle
        new_node = board[row][col-1]
        
        if not new_node.is_margin:
            
            new_c2c = 1 + curr_node.c2c
            
            if new_c2c < new_node.c2c:
                new_node.c2c = new_c2c
                new_node.parent = curr_node
                
            return new_node

    return None


def check_right(board, curr_node):
    
    row = curr_node.cell_location[0]
    col = curr_node.cell_location[1]
    
    # check out of bounds
    if col < 399:
        
        # check if obstacle
        new_node = board[row][col+1]
        
        if not new_node.is_margin:
            
            new_c2c = 1 + curr_node.c2c
            
            if new_c2c < new_node.c2c:
                new_node.c2c = new_c2c
                new_node.parent = curr_node
                
            return new_node

    return None


def check_up_left(board, curr_node):
    
    row = curr_node.cell_location[0]
    col = curr_node.cell_location[1]
    
    # check out of bounds
    if col > 0 and row < 249:
        
        # check if obstacle
        new_node = board[row+1][col-1]
        
        if not new_node.is_margin:
            
            new_c2c = 1.4 + curr_node.c2c
            
            if new_c2c < new_node.c2c:
                new_node.c2c = new_c2c
                new_node.parent = curr_node
                
            return new_node

    return None


def check_up_right(board, curr_node):
    
    row = curr_node.cell_location[0]
    col = curr_node.cell_location[1]
    
    # check out of bounds
    if col < 399 and row < 249:
        
        # check if obstacle
        new_node = board[row+1][col+1]
        
        if not new_node.is_margin:
            
            new_c2c = 1.4 + curr_node.c2c
            
            if new_c2c < new_node.c2c:
                new_node.c2c = new_c2c
                new_node.parent = curr_node
                
            return new_node

    return None


def check_down_left(board, curr_node):
    
    row = curr_node.cell_location[0]
    col = curr_node.cell_location[1]
    
    # check out of bounds
    if col > 0 and row > 0:
        
        # check if obstacle
        new_node = board[row-1][col-1]
        
        if not new_node.is_margin:
            
            new_c2c = 1.4 + curr_node.c2c
            
            if new_c2c < new_node.c2c:
                new_node.c2c = new_c2c
                new_node.parent = curr_node
                
            return new_node

    return None


def check_down_right(board, curr_node):
    
    row = curr_node.cell_location[0]
    col = curr_node.cell_location[1]
    
    # check out of bounds
    if col < 399 and row > 0:
        
        # check if obstacle
        new_node = board[row-1][col+1]
        
        if not new_node.is_margin:
            
            new_c2c = 1.4 + curr_node.c2c
            
            if new_c2c <= new_node.c2c:
                new_node.c2c = new_c2c
                new_node.parent = curr_node
                
            return new_node

    return None


# generate next possible nodes for the current one
# filter out null ones which happens in a bpundary condition
def gen_next_nodes(board, curr_node):

    new_nodes = []

    new_nodes.append(check_up(board, curr_node))
    new_nodes.append(check_down(board, curr_node))
    new_nodes.append(check_left(board, curr_node))
    new_nodes.append(check_right(board, curr_node))
    new_nodes.append(check_up_left(board, curr_node))
    new_nodes.append(check_up_right(board, curr_node))
    new_nodes.append(check_down_left(board, curr_node))
    new_nodes.append(check_down_right(board, curr_node))

    return list(filter(lambda node: node is not None, new_nodes))


# this is the backtracking function
# returns a list of nodes in order to find the solution
def get_solution_path(curr_node):
    solution_path= []
    
    while curr_node:
        solution_path.insert(0, curr_node)
        curr_node = curr_node.parent
        
    return solution_path


# use cv2 in order to draw how the node 
# traversal looks as well as plot the shortest path

def animate(color_map, closed_nodes, solution_path, filename):
    out = cv2.VideoWriter(f'{filename}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, (400, 250))
 
    for node in closed_nodes:
        out.write(np.flipud(update_color_map(node, color_map, [255, 255, 255])))

    for node in solution_path:
        out.write(np.flipud(update_color_map(node, color_map, [0, 255, 0])))
        
    out.release()


def parse_input():

    start_pos = sys.argv[1].split(',')
    start_pos = list(int(n) for n in start_pos)
    
    goal_pos = sys.argv[2].split(',')
    goal_pos = list(int(n) for n in goal_pos)

    try:
        margin = int(sys.argv[3])
    except:
        margin = 5

    return start_pos, goal_pos, margin


# construct the graph of nodes as well as an array of blank rbg values that will represent the board
# perform disjkstra using the node grah, but update the color array as we pop nodes


def main():

    if len(sys.argv) < 3:
            print('Poorly Formatted Input')
            print('Retry with structure \"$ python3 dijkstra-pathplanning-Thomas-Birney.py {start_row},{start_col} {goal_row},{goal_col} margin\"')
            return 0

    start_location, goal_location, margin = parse_input()

    width = 400
    height = 250

    # create the starting board and color map based off of the starting board
    print('Building Board')
    board = create_board(width, height, margin)
    color_map = create_color_map(board)


    if start_location[0] not in range(0, 250) or start_location[1] not in range(0, 400):
        print("Start Location Out Of Bounds")
        return 0

    if goal_location[0] not in range(0, 250) or goal_location[1] not in range(0, 400):
        print("Goal Location Out Of Bounds")
        return 0

    start_node = board[start_location[0]][start_location[1]]
    goal_node = board[goal_location[0]][goal_location[1]]


    if start_node.is_obstacle or start_node.is_margin:
        print('Cannot start in obstacle or obstacle margin')
        return 0
    
    if goal_node.is_obstacle or goal_node.is_margin:
        print('Cannot start in obstacle or obstacle margin')
        return 0


    start_node.c2c = 0

    open_nodes = [start_node]
    closed_nodes = []

    found = False

    print('Searching')
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
            next_possible_nodes = gen_next_nodes(board, curr_node)
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
    
    print('Animating Search Pattern')          
                    
    # back track and animate the search and solution
    solution_path = get_solution_path(curr_node)
    animate(color_map, closed_nodes, solution_path, filename='search')


if __name__ == "__main__":
    try:
        main()
    except:
        pass
