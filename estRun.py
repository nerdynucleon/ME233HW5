from __future__ import division
import numpy as np
import scipy as sp
from scipy.stats import norm

#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

def max_diff_angles(angles):
    angles = np.mod(angles, 2.0*np.pi)
    idx_larger_pi = angles > np.pi
    angles[idx_larger_pi] = (2.0*np.pi) - angles[idx_larger_pi]
    return np.max(angles) - np.min(angles)

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

    particles = internalStateIn[0]
    system = internalStateIn[1]

    x = particles[0,:]
    y = particles[1,:]
    theta = particles[2,:]

    R = system[0,:]
    B = system[1,:]

    # Noise Constants
    slip_scale = 0.1
    std_pedal_speed = 0.5
    std_steering_angle = 0.05

    # Add noise
    pedalSpeed += np.random.normal(scale = std_pedal_speed, size=particles.shape[1])
    steeringAngle += np.random.normal(scale = std_steering_angle, size=particles.shape[1])

    v = pedalSpeed * R * GEAR_RATIO
    v -= np.random.exponential(scale = slip_scale, size = particles.shape[1])

    # Calculate Time Derivatives
    xdot = v * np.cos(theta)
    ydot = v * np.sin(theta)
    thetadot = v * np.tan(steeringAngle) / B 

    # Calculate State Updates
    x = x + dt * xdot
    y = y + dt * ydot
    theta = theta + dt * thetadot

    particles = np.stack((x,y,theta))

    if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
        # Have a valid measurement

        # GPS Characterization Data
        # GPS is zero mean
        # R = np.diag([1.08807010408, 2.98447239424])
        var_meas_0 = 1.08807010408
        var_meas_1 = 2.98447239424

        prob_meas_0 = norm.pdf(measurement[0], (x + 0.5*B*np.cos(theta)), np.sqrt(var_meas_0))
        prob_meas_1 = norm.pdf(measurement[1], (y + 0.5*B*np.sin(theta)), np.sqrt(var_meas_1))
        prob = prob_meas_0 * prob_meas_1

        # Normalize PDF
        prob /= np.sum(prob)

        # Resample
        particles_idx = np.random.choice(np.arange(particles.shape[1]), particles.shape[1], p=prob)
        particles = np.stack((x[particles_idx],y[particles_idx],theta[particles_idx]))

        # Resample System parameters - no roughening
        system = np.stack((R[particles_idx], B[particles_idx]))

        # Roughening
        x = particles[0,:]
        y = particles[1,:]
        theta = particles[2,:]

        zero_mean = np.array([0.0, 0.0, 0.0])

        d = particles.shape[0]
        K = 0.1
        N = particles.shape[1] 

        std_x_rough = K * (N ** (-1/d)) * np.abs(np.max(x) - np.min(x))
        std_y_rough = K * (N ** (-1/d)) * np.abs(np.max(y) - np.min(y))
        std_theta_rough = K * (N ** (-1/d)) * max_diff_angles(theta)
        var_roughening = np.diag([std_x_rough**2.0, std_y_rough**2.0, std_theta_rough**2.0])

        particles += np.random.multivariate_normal(zero_mean, var_roughening, particles.shape[1]).T
        
    #### OUTPUTS ####
    # Update the internal state (will be passed as an argument to the function
    # at next run), must obviously be compatible with the format of
    # internalStateIn:

    internalState = [particles,
                    system
                    ]

    x_est = np.mean(particles[0,:])
    y_est = np.mean(particles[1,:])
    theta_est = np.mean(particles[2,:])

    # DO NOT MODIFY THE OUTPUT FORMAT:
    return x_est, y_est, theta_est, internalState 


