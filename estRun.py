from __future__ import division
import numpy as np
import scipy as sp
from scipy import linalg

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

    state = internalStateIn[0]
    var = internalStateIn[1]

    x = state[0]
    y = state[1]
    theta = state[2]

    R = 0.425
    B = 0.8

    v = pedalSpeed * R * GEAR_RATIO

    # Calculate Time Derivatives
    xdot = v * np.cos(theta)
    ydot = v * np.sin(theta)
    thetadot = v / B * np.tan(steeringAngle)

    # Calculate Variance Updates
    A = np.array([[1.0, 0.0, -dt*v*np.sin(theta)],[0.0, 1.0, dt*v*np.cos(theta)],[0.0, 0.0, 1.0]])
    var = A.dot(var.dot(A.T))

    # Calculate State Updates
    x = x + dt * xdot
    y = y + dt * ydot
    theta = theta + dt * thetadot

    state = np.array([x,y,theta])

    if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
        # Have a valid measurement
        # GPS Characterization Data

        R = np.diag([1.08807010408, 2.98447239424])

        # Try Extended Kalman Filter Update
        H = np.array([[1.0, 0.0, -B/2*np.sin(theta)],[0.0, 1.0, B/2*np.cos(theta)]])
        M = np.eye(2)

        K = var.dot(H.T.dot(sp.linalg.inv(H.dot(var.dot(H.T)) + M.dot(R.dot(M.T)))))
        
        z1_expected = x + 0.5 * B * np.cos(theta)
        z2_expected = y + 0.5 * B * np.sin(theta)
        z_expected = np.array([z1_expected, z2_expected])

        state = np.array([x,y,theta])

        state = state + K.dot(np.array([measurement[0], measurement[1]]) - z_expected)
        var = (np.eye(3) - K.dot(H)).dot(var)

        # weights = [0.5, 0.5]
        # x = np.average([x, measurement[0] - 0.5 * B * np.cos(theta)], weights=weights)
        # y = np.average([y, measurement[1] - 0.5 * B * np.sin(theta)], weights=weights)
        # #weights = [0.95, 0.05]
        # theta = theta #np.average([theta, np.arctan2((measurement[1] - y),(measurement[0] - x))], weights=weights)
        # state = np.array([x,y,theta])

    #### OUTPUTS ####
    # Update the internal state (will be passed as an argument to the function
    # at next run), must obviously be compatible with the format of
    # internalStateIn:

    internalState = [state,
                     var
                    ]

    # DO NOT MODIFY THE OUTPUT FORMAT:
    return x, y, theta, internalState 


