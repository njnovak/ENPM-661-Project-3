from nodes import *

# ''' All functions in this file find moves in the named directions.

# Parameters:
# - current: The current node
# - count: The current iteration

# Returns:
# - out_node: The next node in that direction, with updated parents
# '''


def move_up(current,count):
    out_node = Node(current.name,count,current.priority+1,(current.value[0]-1,current.value[1]))
    return out_node

def move_down(current,count):
    out_node = Node(current.name,count,current.priority+1,(current.value[0]+1,current.value[1]))
    return out_node

def move_right(current,count):
    out_node = Node(current.name,count,current.priority+1,(current.value[0],current.value[1]+1))
    return out_node

def move_left(current,count):
    out_node = Node(current.name,count,current.priority+1,(current.value[0],current.value[1]-1))
    return out_node

def diag_u_right(current,count):
    out_node = Node(current.name,count,current.priority+1.4,(current.value[0]-1,current.value[1]+1))
    return out_node

def diag_b_right(current,count):
    out_node = Node(current.name,count,current.priority+1.4,(current.value[0]+1,current.value[1]+1))
    return out_node

def diag_u_left(current,count):
    out_node = Node(current.name,count,current.priority+1.4,(current.value[0]-1,current.value[1]-1))
    return out_node

def diag_b_left(current,count):
    out_node = Node(current.name,count,current.priority+1.4,(current.value[0]+1,current.value[1]-1))
    return out_node

