import numpy as np

def check_moves(p_queue, next_nodes, canvas,obstacles,explored_names):
    ''' Function to check the next nodes for possible appends to the open list.

    Parameters:
    - p_queue: Node priority queue
    - next_nodes: List of nearby nodes
    - canvas: A mask of the current map
    - obstacles: Representation of the obstacle zones and collision radius
    - explored_names: Names of explored nodes

    Returns:
    p_queue: Modified priority queue
    canvas: Modified canvas
    '''

    for node1 in next_nodes:
        if node1.value not in obstacles and node1.name not in explored_names:
            
            if canvas[node1.value[0]][node1.value[1]] == 1:
                # print("Exploring.")
                p_queue.append(node1)
                canvas[node1.value[0]][node1.value[1]] = 2
            else:
                # print("Try popping.")
                for pri in range(len(p_queue)):
                    if node1.value == p_queue[pri].value and node1.priority < p_queue[pri].priority:
                        # print("Popping.")
                        temp = p_queue.pop(pri)
                        node1.name = temp.name
                        p_queue.append(node1)
    return p_queue,canvas