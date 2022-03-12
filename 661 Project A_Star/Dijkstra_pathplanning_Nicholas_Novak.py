from movements import *
from nodes import *
from map_maker import *
from check_moves import check_moves
import numpy as np
import math
import time


def find_path(origin,goal,obstacles,canvas,outfile,img_array):
    ''' Function to find a path between two points on the provided image map. Will use Dijkstra to find an optimal path.
        Will avoid obstacles, collision radius, and keep a priority queue of nodes (represented as pixels).

    Parameters:
    - origin: Start points
    - goal: End points
    - obstacles: The obstacle space representation
    - canvas: A mask of the current map
    - outfile: The file to which the video will be written
    - img_array: The current map
    
    Returns:
    - explored: A list of all explored nodes during the exploration phase
    '''
    
    orig_node = Node(0,1,0,origin) # First node
    p_queue = [orig_node] # Open list
    explored = [] # Closed list
    current = orig_node # Setup for looping
    count = 2 # Starting name
    explored_names = [] # List of names that have been visited

    

    def is_goal(current,goal):
        ''' Function to check if a node is the goal node

        Parameters:
        - current: The current node
        - goal: The goal node

        Returns: True if the current is the goal, otherwise False
        '''
        if current == goal:
            return True
        return False

    while not is_goal(current.value,goal) and len(p_queue)!=0:
        # Main exploration loop.

        p_queue.sort(key=lambda x: x.priority)
        next_nodes = []
        current = p_queue.pop(0)
        explored.append(current)
        explored_names.append(current.name)
        
        if is_goal(current.value,goal):
            # Secondary break case
            break

        # Below is all the movement and exploration cases for a node given its location
        # ----------------------%%%%%%------------------------ #
        if current.value[1] == 0:
            if current.value[0] == 0:
                # Top Left
                next_nodes.append(move_down(current,count))
                count+=1
                next_nodes.append(move_right(current,count))
                count+=1
                next_nodes.append(diag_b_right(current,count))
                count+=1
                
            elif current.value[0] == len(canvas)-1:
                # Bottom Left
                next_nodes.append(move_up(current,count))
                count+=1
                next_nodes.append(move_right(current,count))
                count+=1
                next_nodes.append(diag_u_right(current,count))
                count+=1
            else:
                # Left
                next_nodes.append(move_right(current,count))
                count+=1
                next_nodes.append(move_up(current,count))
                count+=1
                next_nodes.append(move_down(current,count))
                count+=1
                next_nodes.append(diag_b_right(current,count))
                count+=1
                next_nodes.append(diag_u_right(current,count))
                count+=1


                
            
        elif current.value[1] == len(canvas[0])-1:
            if current.value[0] == 0:
                # Top Right
                next_nodes.append(move_down(current,count))
                count+=1
                next_nodes.append(move_left(current,count))
                count+=1
                next_nodes.append(diag_b_left(current,count))
                count+=1
                
            elif current.value[0] == len(canvas)-1:
                # Bottom Right
                next_nodes.append(move_up(current,count))
                count+=1
                next_nodes.append(move_left(current,count))
                count+=1
                next_nodes.append(diag_u_left(current,count))
                count+=1
            else:
                # Right
                next_nodes.append(move_left(current,count))
                count+=1
                next_nodes.append(move_up(current,count))
                count+=1
                next_nodes.append(move_down(current,count))
                count+=1
                next_nodes.append(diag_b_left(current,count))
                count+=1
                next_nodes.append(diag_u_left(current,count))
                count+=1

        elif current.value[0] == 0:
            # Top
            next_nodes.append(move_left(current,count))
            count+=1
            next_nodes.append(move_right(current,count))
            count+=1
            next_nodes.append(move_down(current,count))
            count+=1
            next_nodes.append(diag_b_left(current,count))
            count+=1
            next_nodes.append(diag_b_right(current,count))
            count+=1

        elif current.value[0] == len(canvas)-1:
            # Bottom
            next_nodes.append(move_up(current,count))
            count+=1
            next_nodes.append(move_left(current,count))
            count+=1
            next_nodes.append(move_right(current,count))
            count+=1
            next_nodes.append(diag_u_left(current,count))
            count+=1
            next_nodes.append(diag_u_right(current,count))
            count+=1

        else:
            # Interior
            next_nodes.append(move_right(current,count))
            count+=1
            next_nodes.append(move_left(current,count))
            count+=1
            next_nodes.append(move_up(current,count))
            count+=1
            next_nodes.append(move_down(current,count))
            count+=1
            next_nodes.append(diag_u_right(current,count))
            count+=1
            next_nodes.append(diag_u_left(current,count))
            count+=1
            next_nodes.append(diag_b_right(current,count))
            count+=1
            next_nodes.append(diag_b_left(current,count))
            count+=1
        # ----------------------%%%%%%------------------------ #

        p_queue,canvas = check_moves(p_queue,next_nodes,canvas,obstacles,explored_names)
        
        img_array[current.value[0]][current.value[1]] = (255,255,255) # Set the explored node to white and record the frame
        outfile.write(img_array)
        
                        
        # print("Current node is: ",current.value,current.priority,current.name,current.parent,count)
    return explored

def backtrack(explored):
    ''' Function to backtrack after exploration
    
    Parameters:
    - explored: A list of explored nodes

    Returns:
    - path: A list of the optimal path of nodes to the goal from the origin
    '''
    initial = explored[0]
    current = explored[-1]
    explored = explored[::-1]
    path = [current]
    while path[-1] != initial: # Main backtrack body
        for i in explored:
            if i.name == current.parent:
                current = i
                break

        # print("Tracking to: ",current.value)
        path.append(current)

    return path

def main(outfile):
    ''' Function to be called in main operation. Calls exploration and backtracking functions as well as interpreting the path.

    Parameters:
    - outfile: The name of the video file.

    Returns:
    - path: A list of the optimal path of nodes to the goal from the origin
    - map_arr: The image of the map, updated
    '''

    obstacle_space = []
    map_arr,canvas = map_maker_default()
    
    c = 0
    r = 0
    for col in map_arr:
        r = 0
        for row in col:
            if row[0] != 255 and row[1] != 255 and row[2] != 0: 
                obstacle_space.append((c,r))
            if row[0] != 0 and row[1] != 0 and row[2] != 255: 
                obstacle_space.append((c,r))
            r+=1
        c+=1
    

    origin = (240,1)
    goal = (50,350) 
    o_x = int(input("What is the x position of the origin (0,0 is lower left)? "))
    o_y = 250-int(input("What is the y position of the origin? "))
    
    g_x = int(input("\n\nWhat is the x position of the goal (0,0 is lower left)? "))
    g_y = 250-int(input("What is the y position of the goal? "))
    origin = (o_y,o_x)
    goal = (g_y,g_x)


    out = find_path(origin,goal,obstacle_space,canvas,outfile,map_arr )
    path = backtrack(out)

    return path,map_arr


if __name__ == "__main__":
    ''' Main operation of the file
    '''
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    outfile = cv2.VideoWriter('map_finders.avi',fourcc,60,(400,250)) # OpenCV Video setup.


    path,img_array = main(outfile)
    path = path[::-1]
    for i in path:
        img_array[i.value[0]][i.value[1]] = (255,0 ,255)
        # Frame padding to allow easier viewing of the path.
        outfile.write(img_array)
        outfile.write(img_array)
        outfile.write(img_array)
        outfile.write(img_array)
    print("Done")

    cv2.imshow("Path", img_array) # Uncomment/comment here to the end to see the end map
    print("Press any key while selecting the image")
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
