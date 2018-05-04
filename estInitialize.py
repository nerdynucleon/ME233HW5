import numpy as np
import scipy as sp
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

np.random.seed(64)
def estInitialize():
    # Fill in whatever initialization you'd like here. This function generates
    # the internal state of the estimator at time 0. You may do whatever you
    # like here, but you must return something that is in the format as may be
    # used by your run() function.
    #

    # Number of Particles
    N = 10000
    # Initialize Particles State
    x = 0.0
    y = 0.0
    theta = np.pi/4.0

    x_std = 7
    y_std = 7
    theta_std = np.pi/2.0

    particle_mean = np.array([x,y,theta])
    particle_var = np.diag([x_std, y_std, theta_std])
    
    # Create State Particles
    particles = np.random.multivariate_normal(particle_mean, particle_var, N)
    particles = particles.T

    # System Parameter Particle Initialization
    R = 0.425
    B = 0.8
    R_DELTA = 0.05 * R
    B_DELTA = 0.1 * B

    system = np.stack((np.random.uniform(R - R_DELTA, R + R_DELTA, N), np.random.uniform(B - B_DELTA, B + B_DELTA, N)))

    # note that there is *absolutely no prescribed format* for this internal state.
    # You can put in it whatever you like. Probably, you'll want to keep the position
    # and angle, and probably you'll remove the color.
    internalState = [particles,
                    system
                     ]

    return internalState

