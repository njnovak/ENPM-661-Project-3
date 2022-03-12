def get_inputs():
    start = input("What is the start point? Provide a list with [x,y,theta]: ")
    goal = input("What is the goal point? Provide a list with [x,y,theta]: ")
    clearance = int(input("What is the clearance radius of the robot? "))
    while step not in range(1,11):
        step = input("What is the distance the robot moves? Enter a number between 1 and 10. ")
    theta2 = input("What is the angle between consecutive actions? ")
    thresh = 1.5

    return([start,goal,clearance,step,theta2,thresh])


