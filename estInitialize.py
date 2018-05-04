import numpy as np
import scipy as sp
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

def estInitialize():
    # Fill in whatever initialization you'd like here. This function generates
    # the internal state of the estimator at time 0. You may do whatever you
    # like here, but you must return something that is in the format as may be
    # used by your run() function.
    #

    #we make the interal state a list, with the first three elements the position
    # x, y; the angle theta; and our favourite color. 
    x = 0.0
    y = 0.0
    theta = np.pi/4.0

    state = np.array([x,y,theta])

    x_var = 2.5
    y_var = 2.5
    theta_var = np.pi/8.0
    
    #R_DELTA = 0.05 * R
    #B_DELTA = 0.1 * B

    var = np.diag([x_var, y_var, theta_var])

    # note that there is *absolutely no prescribed format* for this internal state.
    # You can put in it whatever you like. Probably, you'll want to keep the position
    # and angle, and probably you'll remove the color.
    internalState = [state,
                     var
                     ]

    return internalState

