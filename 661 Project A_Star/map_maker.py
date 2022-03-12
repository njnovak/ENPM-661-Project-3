import cv2
import numpy as np
import math
def get_input(img_array):
    ''' Function to grab an input of the collision radius and modify the map accordingly.

    Parameters:
    - img_array: The current map
    
    Returns:
    - img_array: The updated map
    '''

    border  = int(input("What is the collision radius of the robot? Enter an integer. "))
    
    # Boomerang
    boom_pts_1 = np.array([[36-border,65],[115+abs(round(math.cos(50)*border)),40-abs(round(math.sin(40)*border))],[80+border,70]])
    cv2.fillPoly(img_array,pts = [boom_pts_1],color = (0,0,255))
    boom_pts_2 = np.array([[36-border,65],[105+abs(round(math.cos(70)*border)),150+border],[80+border,70]])
    cv2.fillPoly(img_array,pts = [boom_pts_2],color = (0,0,255))

    # Circle
    cv2.circle(img_array,(300, 65),40+border,(0,0,255),cv2.FILLED)

    # Hexagon
    hex_points_1 = np.array([[200,110-border],[165-abs(round(math.cos(60)*border)),130-abs(round(math.sin(600)*border))],[165-abs(round(math.cos(60)*border)),170+abs(round(math.sin(60)*border))],[200 ,190+border]])
    cv2.fillPoly(img_array,pts = [hex_points_1],color = (0,0,255))
    hex_points_2 = np.array([[200,110-border],[235+abs(round(math.cos(60)*border)),130-abs(round(math.sin(60)*border))],[235+abs(round(math.cos(60)*border)),170+abs(round(math.sin(60)*border))],[200,190+border]])
    cv2.fillPoly(img_array,pts = [hex_points_2],color = (0,0,255))
    return img_array
def map_maker_default():
    ''' Default map maker function. Will develop the provided map as a numpy array for OpenCV purposes.
        Also makes a mask of unexplored nodes
    
    Returns:
    - img_array: The updated map
    - canvas: The map mask
    '''
    blank_canvas = []
    
    img_array = np.zeros((250,400,3), dtype = np.uint8)
    for col in range(400):
        blank_canvas.append([])
        for row in range(250): 
            blank_canvas[col].append([])
            blank_canvas[col][row] = [0,0,0]
    img_array = get_input(img_array)

    # Boomerang
    boom_pts_1 = np.array([[36,65],[115,40],[80,70]])
    cv2.fillPoly(img_array,pts = [boom_pts_1],color = (255,255,0))
    boom_pts_2 = np.array([[36,65],[105,150],[80,70]])
    cv2.fillPoly(img_array,pts = [boom_pts_2],color = (255,255,0))

    
    # Circle
    cv2.circle(img_array,(300, 65),40,(255,255,0),cv2.FILLED)

    #  Hexagon
    hex_points_1 = np.array([[200,110],[165,130],[165,170],[200,190]])
    cv2.fillPoly(img_array,pts = [hex_points_1],color = (255,255,0))
    hex_points_2 = np.array([[200,110],[235,130],[235,170],[200,190]])
    cv2.fillPoly(img_array,pts = [hex_points_2],color = (255,255,0))


    obstacles = np.where((img_array[:,:,0]==255) & (img_array[:,:,1]==255) & (img_array[:,:,2]==0))
    canvas = np.zeros((250,400))
    for col in range(len(canvas)):
        for row in range(len(canvas[0])):
            canvas[col][row] = 1
    canvas[obstacles] = -1
    cv2.imshow("Map",img_array) # Uncomment/comment these 3 lines to show the map upon running
    print("Press any key while selecting the image")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return(img_array,canvas)
    
