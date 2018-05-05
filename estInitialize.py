import numpy as np
import scipy as sp
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

np.random.seed(64)
def estInitialize():
    # Author - Alexandre Chong
    # Fill in whatever initialization you'd like here. This function generates
    # the internal state of the estimator at time 0. You may do whatever you
    # like here, but you must return something that is in the format as may be
    # used by your run() function.

    # Number of Particles
    N = 10000
    # Initialize Particles State
    x = 0.0
    y = 0.0
    theta = np.pi/4.0

    x_var = 2**2
    y_var = 2**2
    theta_var = (np.pi/4.0)**2

    particle_mean = np.array([x,y,theta])
    particle_var = np.diag([x_var, y_var, theta_var])
    
    # Create State Particles
    particles = np.random.multivariate_normal(particle_mean, particle_var, N)
    particles = particles.T

    # System Parameter Particle Initialization
    R = 0.425
    B = 0.8
    R_DELTA = 0.05 * R
    B_DELTA = 0.1 * B

    system = np.stack((np.random.uniform(R - R_DELTA, R + R_DELTA, N), np.random.uniform(B - B_DELTA, B + B_DELTA, N)))

    # Create Internal State Data Structure
    internalState = [particles,
                    system
                     ]

    return internalState

