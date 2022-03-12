class Node:
    
    def __init__(self, parent, name, priority, value):
        ''' Custom class object to represent each map pixel.

        Paramters:
        - parent: The parent node's name (int)
        - name: This node's unique name (int)
        - priority: The current cost to come of the node in the queue (float)
        - value: The position of the pixel in the map, in (y,x) form (tuple(int,int))
    
        '''
        self.parent = parent
        self.name = name
        self.priority = priority # Cost to Come
        self.value = value

