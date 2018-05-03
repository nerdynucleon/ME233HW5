import numpy as np
import scipy as sp
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

def estRun(time, dt, internalStateIn, steeringAngle, pedalSpeed, measurement):
    # In this function you implement your estimator. The function arguments
    # are:
    #  time: current time in [s] 
    #  dt: current time step [s]
    #  internalStateIn: the estimator internal state, definition up to you. 
    #  steeringAngle: the steering angle of the bike, gamma, [rad] 
    #  pedalSpeed: the rotational speed of the pedal, omega, [rad/s] 
    #  measurement: the position measurement valid at the current time step
    #
    # Note: the measurement is a 2D vector, of x-y position measurement.
    #  The measurement sensor may fail to return data, in which case the
    #  measurement is given as NaN (not a number).
    #
    # The function has four outputs:
    #  est_x: your current best estimate for the bicycle's x-position
    #  est_y: your current best estimate for the bicycle's y-position
    #  est_theta: your current best estimate for the bicycle's rotation theta
    #  internalState: the estimator's internal state, in a format that can be understood by the next call to this function

    # Example code only, you'll want to heavily modify this.
    # this internal state needs to correspond to your init function:

    ## DEFINE CONSTANTS
    GEAR_RATIO = 5.0
    RADIUS = 0.425
    RADIUS_DELTA = 0.05 * RADIUS
    BASE = 0.8
    BASE_DELTA = 0.1 * BASE

    x = internalStateIn[0]
    y = internalStateIn[1]
    theta = internalStateIn[2]

    v = pedalSpeed * RADIUS * GEAR_RATIO

    # Calculate Time Derivatives
    xdot = v * np.cos(theta)
    ydot = v * np.sin(theta)
    thetadot = v / BASE * np.tan(steeringAngle)

    # Calculate State Updates
    theta = theta + dt * thetadot
    x = x + dt * xdot
    y = y + dt * ydot

    if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
        # have a valid measurement
        weights = [0.75, 0.25]
        x = np.average([x, measurement[0] - 0.5 * BASE * np.cos(theta)], weights=weights)
        y = np.average([y, measurement[1] - 0.5 * BASE * np.sin(theta)], weights=weights)
        theta = theta

    #### OUTPUTS ####
    # Update the internal state (will be passed as an argument to the function
    # at next run), must obviously be compatible with the format of
    # internalStateIn:
    internalState = [x,
                     y,
                     theta
                    ]

    # DO NOT MODIFY THE OUTPUT FORMAT:
    return x, y, theta, internalState 


