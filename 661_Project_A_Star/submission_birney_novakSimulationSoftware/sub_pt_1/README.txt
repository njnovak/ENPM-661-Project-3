Instructions:

To run this program, simply run the .py file, making sure all libraries are installed.
You are able to change the start and goal locations, as well as the clearance for the robot. 
These are editable in lines 390 to 394 of the code. When entering these coordinates they are in [y, x, theta]
format. Also 1 meter in real life is 20 cells of the map. So take your coordinates in meters and multiply is by 20
to get the appropriateboard coordinate. So a point at x =0.25, y = 0.25 would actually be x = 5, y = 5. Just make
sure the start and end points are within the bounds and unoccupied by an obstacle

Thetas are input as degrees

The rpms are non editable because for some reason backtracking hangs forever when I do this. SO instead we have set rpms at 3 and 7

Also the goal point was placed closer to the start purely for computations times sake. 
I encourage you to pick a valid goal point if you want to see it traverse the map completely.

Any other inquires can be sent to tbirney@umd.edu
Operation can be seen in the included test.avi file
Code can be found at: https://github.com/njnovak/ENPM-661-Project-3
See each file's comments for further explanations
