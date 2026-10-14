'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 1A of Niti Vahan (NV) Theme.
*
*  Filename:         ackermann_steering.py
*  Created:          2026
*  Last Modified:
*  Author:           e-Yantra Team
*
*  You are ONLY allowed to write your code inside the block marked
*  "ADD YOUR IMPLEMENTATION HERE". Do not change anything outside it - the
*  evaluation script relies on the rest of this file staying as it is.
*
*****************************************************************************************
'''

# Team ID:          1568
# Author List:      Priyansh Miyani
# Filename:         ackermann_steering.py
# Functions:        ackermann_wheel_angles
# Global variables: None


####################### IMPORT MODULES #######################
import math
import numpy as np
##############################################################


#################### VEHICLE CONSTANTS #######################
WHEELBASE = 0.120           # L: distance between front and rear axle centrelines
TRACK_WIDTH = 0.110         # W: distance between left and right wheel centre
WHEEL_OFFSET = 0.0275       # O: distance between kingpin axis and wheel centre.
##############################################################


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

def radius_from_angle(delta):
  r=WHEELBASE/math.tan(abs(delta))#calculate the radius 
  return r

def ackermann_wheel_angles(delta):
    '''
    Purpose:
    ---
    Convert a single virtual steering angle into the two real front-wheel
    angles, per the Ackermann geometry.

    Input Arguments:
    ---
    `delta` :   [ float ]
        Steering angle of the virtual centred front wheel, in radians.

    Returns:
    ---
    `left_angle`  : [ float ]
    `right_angle` : [ float ]
        The two real front-wheel steering angles, in radians, using the
        same sign convention as delta.

    REMEMBER:
    ---
    WHEEL_OFFSET changes the effective half-track width inside each wheel's triangle.
        '''
    if delta == 0: #checking if the desired steering angle is zero (straight)
        return(0,0)
    elif delta>0: #checking for a left turn
        R= radius_from_angle(delta)
        left=R-(TRACK_WIDTH-2*WHEEL_OFFSET)/2 #calculating radius for left wheel to turn left
        right=R+(TRACK_WIDTH-2*WHEEL_OFFSET)/2#calculating radius for right wheel to turn left
        
        left_wheel_angle = math.atan(WHEELBASE/left)#calculating the angle for left wheel to turn left
        right_wheel_angle = math.atan(WHEELBASE/right)#calculating the angle for right wheel to turn left
        
        return left_wheel_angle,right_wheel_angle
    else : #otherwise, it is a right turn 
        R = radius_from_angle(delta)
        left=R+(TRACK_WIDTH-2*WHEEL_OFFSET)/2 #calculating radius for left wheel to turn right
        right=R-(TRACK_WIDTH-2*WHEEL_OFFSET)/2 #calculating radius for right wheel to turn right
        
        left_wheel_angle = -math.atan(WHEELBASE/left)#calculating the angle for left wheel to turn right
        right_wheel_angle = -math.atan(WHEELBASE/right)#calculating the angle for right wheel to turn right
        
        return left_wheel_angle,right_wheel_angle


##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

if __name__ == "__main__":

    test_angles = np.arange(-0.35, 0.35, 0.05)

    for d in test_angles:
        left, right = ackermann_wheel_angles(d)
        print(f"delta={d}  ->  left={left}, right={right}")
