# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 22:12:12 2024

@author: 59939, 59249, 60461
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import multiprocessing as mp
from functools import partial
import time
from sklearn.metrics import mean_squared_error

# CONSTANTS
earth_mass = 5.97219 * 10**24
grav_cte = 6.67430 * 10**(-11)
R_earth = 6.371 * 10**6
g_sea = 10.


#%%    
def gravity(h):
    """
    Calculates the gravitational acceleration for a given altitude.

    Parameters
    ----------
    h : float
        Altitude above the Earth's surface.

    Returns
    -------
    g_vector : numpy array
        Vector containing the gravitational acceleration in 2D, with components 
        [0, -g].
    """
    
    g = (grav_cte * earth_mass / ((R_earth + h)**2))
    g_vector = np.array([0, -g])
    
    return g_vector

#%%   
def positive_f(t, f):
    """
    Ensures non-negative values for a function f(t).

    Parameters
    ----------
    t : scalar
        Input to the function f.
    f : callable
        Function that takes t as input

    Returns
    -------
    y : callable
        Function with modified values - negative values are replaced with 0.
    """
    
    y = f(t)
    if np.any(y < 0):
        y[y < 0] = 0
    
    return y
    

def ad_cubic_spline(filename, plot = False):    
    """
    Performs cubic spline interpolation on data from a file.

    Parameters
    ----------
    filename : str
        Name of the file containing data in two columns (x, y).
    plot : bool, optional
        If True, plot the data points and the cubic spline interpolation and 
        print the RMSE value. Default is False.

    Returns
    -------
    cs_modified : callable
        A function that represents the adaptive cubic spline interpolation of 
        the data.
    """
    
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    cs_original = CubicSpline(x, y)
    cs_modified = partial(positive_f, f = cs_original)
    
    if plot:
        x_fine = np.linspace(-1000, 150000, 10000)
        y_fine = cs_modified(x_fine)
    
        plt.scatter(x, y, label='Data points', color='red', s=8)
        plt.plot(x_fine, y_fine, label='Cubic Spline', linewidth=0.8)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('Cubic Spline Interpolation')
        
        plt.show()
        
        y_data = [cs_modified(h) for h in x]
        rmse = np.sqrt(mean_squared_error(y, y_data))
        print(f'RMSE for Cubic Spline Fitting: {rmse}')
    
    return cs_modified


#%%    
def exponential_function(x, a, b):
    """
    Computes the exponential function a * exp(b*x).

    Parameters
    ----------
    x : scalar
        Input value for the exponential function.
    a : scalar
        Amplitude parameter.
    b : scalar
        Exponential growth rate parameter.

    Returns
    -------
    y : scalar
        Output value of the exponential function a * exp(b*x).
    """
    
    return a * np.exp(b*x)



def ad_exp(filename, plot = False):
    """
    Performs exponential curve fitting on data loaded from a file.

    Parameters
    ----------
    filename : str
        Name of the file containing data in two columns (x, y).
    plot : bool, optional
        If True, plot the data points and the fitted exponential curve and print
        the RMSE value. Default is False.

    Returns
    -------
    fitted_exp : callable
        A function representing the fitted exponential curve.
    """
    
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    
    initial_guess = [1, -0.001]
    params, covariance = curve_fit(exponential_function, x, y, p0=initial_guess)
    
    param1, param2 = params
    fitted_exp = partial(exponential_function, a=param1, b=param2)
    
    if plot:
        x_fine = np.linspace(-1000, 150000, 10000)
        y_fine = fitted_exp(x_fine)
    
        plt.scatter(x, y, label='Data points', color='red', s=8)
        plt.plot(x_fine, y_fine, label='Fitted Exponential Curve', linewidth=0.8)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('Exponential Curve Fitting')
        
        plt.show()
        
        y_data = [fitted_exp(h) for h in x]
        rmse = np.sqrt(mean_squared_error(y, y_data))
        print(f'RMSE for Exponencial Fitting: {rmse}')
            
    return fitted_exp


#%%  
def nasa(h): 
    """
    Calculates the air density based on altitude using NASA's atmospheric model.
 
    Parameters
    ----------
    h : float
        Altitude in meters above sea level.

    Returns
    -------
    rho : float
        Air density in kg/m^3 at the given altitude h.
    """
    
    if h > 25000: #upper_stratosphere
        T = -131.21 + 0.00299 * h
        p = 2.488 * ((T + 273.1) / 216.6) ** -11.388
        rho = p / (0.2869 * (T + 273.1))
        return rho
    
    elif 11000 < h <= 25000: #lower_stratosphere
        T = -56.46
        p = 22.65 * np.exp(1.73 - 0.000157 * h)
        rho = p / (0.2869 * (T + 273.1))
        return rho
    
    else: #troposphere
        T = 15.04 - 0.00649 * h
        p = 101.29 * ((T + 273.1) / 288.08) ** 5.256
        rho = p / (0.2869 * (T + 273.1))
        return rho
    


def ad_nasa(filename, plot = False):
    """
    Plots NASA's atmospheric model for air density and the air density data
    loaded from a file.

    Parameters
    ----------
    filename : str
        Name of the file containing data in two columns (x, y).
    plot : bool, optional
        If True, plot the data points and the fitted exponential curve and print
        the RMSE value. Default is False.

    Returns
    -------
    nasa : callable
        A function representing the NASA atmospheric model.
    """
    
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    
    if plot:
        x_fine = np.linspace(-1000, 150000, 10000)
        y_fine = [nasa(h) for h in x_fine]
    
        plt.scatter(x, y, label='Data points', color='red', s=8)
        plt.plot(x_fine, y_fine, label='Fitted Curve', linewidth=0.8)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('NASA Model Fitting')
        
        plt.show()
        
        y_data = [nasa(h) for h in x]
        rmse = np.sqrt(mean_squared_error(y, y_data))
        print(f'RMSE for NASA Model: {rmse}')
    
    return nasa


#%% 
def ad_derivative(f, point, param):
    """
    Approximates the derivative of the function `f` at a given `point` 
    using the finite differences method with a step size defined in `param`.

    Parameters
    ----------
    f : function
        The function for which the derivative is to be calculated.
    point : float
        The point at which the derivative is to be approximated.
    param : dict
        Dictionary containing parameters including 'deriv_dp', the step size 
        for finite differences method.

    Returns
    -------
    d : float
        The approximate derivative of the function `f` at the given `point`.
    """
    
    p1 = point + param['deriv_dp']
    p2 = point - param['deriv_dp']
    d = (f(p1) - f(p2)) / (p1 - p2)
    
    return d


def gravity_derivative(h):
    """
    Calculates the derivative of the gravitational force with respect to 
    height `h` above the Earth's surface.
    
    Parameters
    ----------
    h : float
        The height above the Earth's surface at which the derivative is to be 
        calculated.

    Returns
    -------
    d : float
        The derivative of the gravitational force with respect to height `h`.
    """
    
    return 2 * grav_cte * earth_mass / ((R_earth + h)**3)


#%%  
def reentrySlope_before_parachutes(v, p, param, ad_function): 
    """
    Calculates the slope of the velocity vector - the acceleration - during 
    re-entry before parachute deployment.

    Parameters
    ----------
    v : numpy array
        Velocity vector [vx, vy].
    p : numpy array
        Position vector [px, py].
    param : dict
        Dictionary containing parameters including mass, area, and drag/lift 
        coefficients of the space module.
    ad_function : function
        Function to calculate air density based on altitude.

    Returns
    -------
    slope : numpy array
        Acceleration during re-entry.
    """
    
    g = gravity(p[1])
    ad = ad_function(p[1])
    
    v_norm = np.linalg.norm(v)
    slope = g + (0.5 * ad * param['A_sm'] / param['m_sm']) * ( - param['Cd'] * 
            v_norm * v + param['Cl'] * v_norm**2 * np.array([0, 1]))
    
    return slope


#%% 
def reentrySlope_after_parachutes(v, p, param, ad_function): 
    """
    Calculates the slope of the velocity vector - the acceleration - during 
    re-entry AFTER parachute deployment.

    Parameters
    ----------
    v : numpy array
        Velocity vector [vx, vy].
    p : numpy array
        Position vector [px, py].
    param : dict
        Dictionary containing parameters including mass, area, and drag/lift 
        coefficients of the space module.
    ad_function : function
        Function to calculate air density based on altitude.

    Returns
    -------
    slope : numpy array
        Acceleration during re-entry.
    """
    
    
    g = gravity(p[1])
    ad = ad_function(p[1])
    
    v_norm = np.linalg.norm(v)
    slope = g - (0.5 * ad * v_norm * v / param['m_sm']) * (param['A_sm'] * 
            param['Cd'] + param['A_p'] * param['Cdp'])
    
    return slope


#%%    
def residual(u, param, dt, k, ad, slope):
    """
    Calculates the residual function(s) for a system of equations based on the 
    given parameters.
    
    Parameters
    ----------
    u : array
        The current state vector [u1, u2, u3, u4], where:
        u1 : float - current position in x-direction
        u2 : float - current position in y-direction
        u3 : float - current velocity in x-direction
        u4 : float - current velocity in y-direction
    param : dict
        Parameters required by the slope function.
    dt : float
        The time step size.
    k : array
        The previous state vector [xk, yk, vxk, vyk].
    ad : function
        Air density function required by the slope function.
    slope : function
        The function that calculates the acceleration.
        
    Returns
    -------
    np array
        The array of residuals [r1, r2, r3, r4], where:
        r1 : float - residual for position in x-direction
        r2 : float - residual for position in y-direction
        r3 : float - residual for velocity in x-direction
        r4 : float - residual for velocity in y-direction
    """

    u1, u2, u3, u4 = u
    xk, yk, vxk, vyk = k
    
    uv = np.array([u3, u4])
    up = np.array([u1, u2])
    acc = slope(uv, up, param, ad)
    
    f1 = u3 
    f2 = u4
    f3 = acc[0]  
    f4 = acc[1]
    
    r1 = (1/dt) * (u1-xk) - f1
    r2 = (1/dt) * (u2-yk) - f2
    r3 = (1/dt) * (u3-vxk) - f3
    r4 = (1/dt) * (u4-vyk) - f4
    
    return np.array([r1, r2, r3, r4])


def jacobian(u, param, dt, ad, pc):
    """
    Calculates the Jacobian matrix for a system of differential equations based 
    on the given parameters.

    Parameters
    ----------
    u : array
        The current state vector [u1, u2, u3, u4].
    param : dict
        Parameters required for the Jacobian calculation.
    dt : float
        The time step size.
    ad : function
        Function to calculate air density as a function of height.
    pc : str
        Parameter to differentiate between reentry phases 
        ('reentrySlope_before_parachutes' or 'reentrySlope_after_parachutes').

    Returns
    -------
    np array
        The Jacobian matrix J of shape (4, 4), where J[i][j] represents 
        the partial derivative of residual[i] with respect to u[j].
    """
    
    u1, u2, u3, u4 = u
    
    uv = np.array([u3, u4])
    uv_norm = np.linalg.norm(uv)
    
    air_density = ad(u2)
    ad_deriv = ad_derivative(ad, u2, param)
    g_deriv = gravity_derivative(u2)
    
    I = np.identity(4)
    j = np.zeros((4, 4), dtype = float)
    
    if pc == reentrySlope_before_parachutes:
        
        j[0][2] = 1
        
        j[1][3] = 1
        
        j[2][1] = - (1/2) * (1/param['m_sm']) * param['A_sm'] * param['Cd'] * \
                  uv_norm * u3 * ad_deriv
                  
        j[2][2] = - (1/2) * (1/param['m_sm']) * param['A_sm'] * param['Cd'] * \
                 air_density * ((1/uv_norm) * u3**2 + uv_norm)
        
        j[2][3] = - (1/2) * (1/param['m_sm']) * param['A_sm'] * param['Cd'] * \
                 air_density * (1/uv_norm) * u3 * u4
        
        j[3][1] = g_deriv + (1/2) * (1/param['m_sm']) * param['A_sm'] * \
                  (-param['Cd'] * uv_norm * u4 + param['Cl'] * 
                   uv_norm ** 2) * ad_deriv
        
        j[3][2] = - (1/2) * (1/param['m_sm']) * param['A_sm'] * air_density * \
                  (param['Cd'] * u3 * u4 * (1/uv_norm) - param['Cl'] * 2 * u3)
        
        j[3][3] = - (1/2) * (1/param['m_sm']) * param['A_sm'] * air_density * \
                  (param['Cd'] * ((1/uv_norm) * u4**2 + uv_norm) -
                   param['Cl'] * 2 * u4)
        
    if pc == reentrySlope_after_parachutes:
        
        j[0][2] = 1
        
        j[1][3] = 1
        
        j[2][1] = - (1/2) * (1/param['m_sm']) * uv_norm * (param['A_sm'] * \
                  param['Cd'] + param['A_p'] * param['Cdp']) * u3 * ad_deriv
                  
        j[2][2] = - (1/2) * air_density * (1/param['m_sm']) * (param['A_sm'] \
                  * param['Cd'] + param['A_p'] * param['Cdp']) * ((1/uv_norm) \
                  * u3**2 + uv_norm)
        
        j[2][3] = - (1/2) * air_density * (1/param['m_sm']) * (param['A_sm'] \
                  * param['Cd'] + param['A_p'] * param['Cdp']) * (1/uv_norm) \
                  * u3 * u4
        
        j[3][1] = g_deriv - (1/2) * (1/param['m_sm']) * uv_norm * \
                  (param['A_sm'] * param['Cd'] + param['A_p'] * param['Cdp']) \
                  * u4 * ad_deriv
        
        j[3][2] = - (1/2) * air_density * (1/param['m_sm']) * (param['A_sm'] \
                  * param['Cd'] + param['A_p'] * param['Cdp']) * (1/uv_norm) \
                  * u3 * u4
        
        j[3][3] = - (1/2) * air_density * (1/param['m_sm']) * (param['A_sm'] \
                  * param['Cd'] + param['A_p'] * param['Cdp']) * ((1/uv_norm) \
                  * u4**2 + uv_norm)    
    
    J = (1/dt) * I - j
    
    return J


#%% 
def newton_raphson(guess, dt, param, res, jac): 
    """
    Performs Newton-Raphson iteration to solve a system of nonlinear equations.

    Parameters
    ----------
    guess : array
        Initial guess for the solution vector.
    dt : float
        Time step size.
    param : dict
        Parameters required by the residual function and Jacobian function and 
        for convergence conditions of the method.
    res : function
        Function that computes the residual (system of nonlinear equations).
    jac : function
        Function that computes the Jacobian matrix of `res` (partial derivatives 
        of `res` w.r.t. `u`).

    Returns
    -------
    np array
        The solution vector `u` that satisfies `res(u, param, dt) ≈ 0` within 
        the specified tolerance.
    """
    
    u = guess
    tolerance = param['newt_tol'] 
    
    i = 0
    while np.linalg.norm(res(u, param, dt)) > tolerance and \
          i < param['newt_max_it']: 
        
        res_mx = res(u, param, dt)
        jac_mx = jac(u, param, dt)
        
        J_inv = np.linalg.pinv(jac_mx)
        u = u - np.dot(J_inv, res_mx) 

        i += 1
        
    return u 


#%% 
def backward_euler(v, p, dt, param, acc_slope, ad_function):
    """
    Performs a single step of the backward Euler method for solving differential 
    equations.

    Parameters
    ----------
    v : array
        Current velocity vector [vx, vy].
    p : array
        Current position vector [x, y].
    dt : float
        Time step size.
    param : dict
        Parameters required by the residual function `residual` and Jacobian 
        function `jacobian`.
    acc_slope : function
        Function to compute acceleration based on velocity and position.
    ad_function : function
        Function to compute air density as a function of height.

    Returns
    -------
    v : array
        Updated velocity vector [vx, vy].
    p : array
        Updated position vector [px, py].
    a : array
        Updated acceleration vector [ax, ay].
    """
       
    uk = np.concatenate((p, v))
    
    res = partial(residual, k = uk, ad = ad_function, slope = acc_slope)
    jac = partial(jacobian, ad = ad_function, pc = acc_slope)
    
    init_guess = uk
    uk_new = newton_raphson(init_guess, dt, param, res, jac)
    
    split_uk = np.split(uk_new, 2)
    
    p = split_uk[0]
    v = split_uk[1]
    
    a = acc_slope(v, p, param, ad_function)
    
    return v, p, a


#%%   
def reentry(param, dt, ad_function, v0, alpha0, method):
    """
    Simulates the re-entry trajectory of a spacecraft.
    
    - Initializes velocity, position, aceleration and theta arrays.
    - Sets initial conditions for velocity and altitude.
    - Iteratively computes the position, velocity, and acceleration at each 
      time step using the provided numerical method.
    - Adjusts for parachute deployment based on altitude and velocity conditions.
    - Computes the ground track angle to account for Earth's curvature.
    - Trims the arrays to remove unused elements and calculates the projected 
      distance on the surface using the final (cummulative) theta value.

    Parameters
    ----------
    param : dict
        Dictionary containing parameters including mass, area and drag/lift 
        coeficients of the space module
    dt : float
        Time step used in the re-entry simulation.
    ad_function : function
        Function that computes air density as a function of altitude.
    v0 : float
        Initial velocity magnitude.
    alpha0 : float
        Initial angle of attack in radians.
    method : function
        Numerical integration method (e.g., Euler, RK2) to use for simulation.

    Returns
    -------
    v : numpy array
        Array of velocity vectors [vx, vy] over time during re-entry.
    p : numpy array
        Array of position vectors [x, y] over time during re-entry.
    a : numpy array
        Array of acceleration vectors [ax, ay] over time during re-entry.
    proj_dist : float
        Projected distance on the Earth's surface at the end of re-entry.
    theta : numpy array
        Array of cummulative theta values over time during re-entry.
    """
    
    v = np.zeros((param['size'], 2), dtype = float)
    p = np.zeros((param['size'], 2), dtype = float)
    a = np.zeros((param['size'], 2), dtype = float)
    theta = np.zeros(param['size'], dtype = float)
    
    pc = 0
    slope = reentrySlope_before_parachutes
    
    v[0][0] = v0 * np.cos(alpha0)
    v[0][1] = - v0 * np.sin(alpha0)
    
    p[0][1] = param['h']
    
    a[0] = slope(v[0], p[0], param, ad_function)
    
    i = 1
    while p[i-1][1] > 0 and i < (param['size'] - 1):
        
        if p[i-1][1] <= param['h_p'] and np.linalg.norm(v[i-1]) <= param['v_p'] \
            and pc == 0:
            pc = 1
            slope = reentrySlope_after_parachutes
        
        v[i], p[i], a[i] = method(v[i-1], p[i-1], dt, param, slope, ad_function)
        
        theta[i] = (p[i][0] - p[i-1][0]) / (R_earth + p[i][1]) + theta[i-1]
        
        i += 1
    
    last_nonzero_index = np.max(np.nonzero(p[:, 1]))
    v = v[:last_nonzero_index + 1]
    p = p[:last_nonzero_index + 1]
    a = a[:last_nonzero_index + 1]
    theta = theta[:last_nonzero_index + 1]
    proj_dist = R_earth * theta[last_nonzero_index]
             
    return v, p, a, proj_dist, theta
 
    
#%%  
def plot_data(dt, v, p, a, proj_dist, theta): 
    """
    Plots the velocity, altitude, and acceleration data over time.
    Plots the trajectory.

    Parameters
    ----------
    dt : float
        Time step used in the re-entry simulation.
    v : numpy array
        Array of velocity vectors [vx, vy] over time during re-entry.
    p : numpy array
        Array of position vectors [x, y] over time during re-entry.
    a : numpy array
        Array of acceleration vectors [ax, ay] over time during re-entry.
    proj_dist : float
        Projected distance on the Earth's surface at the end of re-entry.
    theta : numpy array
        Array of cummulative theta values over time during re-entry.
    """
    
    v_norms = np.linalg.norm(v, axis=1)
    a[:, 1] += g_sea
    g_values = np.linalg.norm(a, axis=1)
    g_values /= g_sea
    py = p[:, 1]
    x = np.arange(0, len(v_norms) * dt, dt)
    trajectory = np.zeros((len(py), 2), dtype = float)
    trajectory[:, 0] = (py+R_earth) * np.sin(theta)
    trajectory[:, 1] = (py+R_earth) * np.cos(theta)
    
    plt.plot(x, v_norms)
    plt.xlabel('t (s)')
    plt.ylabel('v (m/s)')
    plt.title('Velocity vs Time')
    plt.grid(True)
    plt.show()
    
    plt.plot(x, py)
    plt.xlabel('t (s)')
    plt.ylabel('h (m)')
    plt.title('Altitude vs Time')
    plt.grid(True)
    plt.show()
    
    plt.plot(x, g_values)
    plt.xlabel('t (s)')
    plt.ylabel('a (m/s²)')
    plt.title('Acceleration vs Time')
    plt.grid(True)
    plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(trajectory[:, 0], trajectory[:, 1], color='red')
    ax.set_title('Trajectory')
    ax.set_ylim(4.8e6, 6.8e6)
    ax.set_xlim(0, 4.2e6)
    radius = R_earth
    circle = plt.Circle((0, 0), radius, color='blue', alpha=1)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    plt.show()
    

#%%   
def analyse_data(dt, v, p, a, proj_dist, theta, param):
    """
    Analyzes the re-entry data to determine if the final velocity and maximum 
    acceleration meet specified criteria and if the projected distance falls 
    within a given range.
    
    Parameters
    ----------
    dt : float
        Time step used in the re-entry simulation.
    v : numpy array
        Array of velocity vectors [vx, vy] over time during re-entry.
    p : numpy array
        Array of position vectors [x, y] over time during re-entry.
    a : numpy array
        Array of acceleration vectors [ax, ay] over time during re-entry.
    proj_dist : float
        Projected distance on the Earth's surface at the end of re-entry.
    theta : numpy array
        Array of cumulative theta values over time during re-entry.
    param : dict
        Dictionary containing parameters including:
        - 'last_v' (float): Maximum allowable final velocity.
        - 'max_g' (float): Maximum allowable g-force.
        - 'dist_min' (float): Minimum allowable projected distance.
        - 'dist_max' (float): Maximum allowable projected distance.
    
    Returns
    -------
    check: list
        - check[0] = 0 if the final velocity is below 'last_v', 1 otherwise
        - check[1] = 0 if the maximum g-force is below 'max_g',  1 otherwise
        - check[2] = 0 if the projected distance is within 'dist_min' and 
        'dist_max',  1 otherwise
    max_g : float
        Maximum g-force experienced during re-entry.
    last_v : float
        Final velocity magnitude.
    proj_dist : float
        Projected distance on the Earth's surface at the end of re-entry.
    """
    
    v_norms = np.linalg.norm(v, axis=1)
    points = len(v_norms)
    last_v = v_norms[points-1]
    
    a[:, 1] += g_sea
    g_values = np.linalg.norm(a, axis=1)
    g_values /= g_sea
    max_g = np.max(g_values)
    
    check = [1,1,1]
    if last_v < param['last_v']:
        check[0] = 0
    if max_g < param['max_g']:
        check[1] = 0
    if param['dist_min'] <= proj_dist <= param['dist_max']:
        check[2] = 0
    
    return check, max_g, last_v, proj_dist 
    
    
#%%    
def many_pairs_parallel(args):
    """
    Simulates a single re-entry scenario with given initial velocity and angle, 
    and returns the results indicating whether the scenario meets the specified 
    criteria.
    Used for parallel processing.

    Parameters
    ----------
    args : tuple
        Tuple containing the following elements:
        - param (dict): Dictionary containing physical parameters required for 
                        the simulation, such as mass, cross-sectional area, 
                        drag coefficient, lift coefficient, etc.
        - dt (float): Time step used in the re-entry simulation.
        - ad_function (function): Function that computes air density as a 
                                  function of altitude.
        - v_init (float): Initial velocity to test.
        - alpha_init (float): Initial angle (in degrees) to test.
        - method (function): Numerical integration method (e.g., Euler, RK2) 
                             used to compute the trajectory.

    Returns
    -------
    v_init : float
        Initial velocity used in the simulation.
    alpha_init : float
        Initial angle (in degrees) used in the simulation.
    color : str
        Color string indicating whether the re-entry scenario meets the 
        specified criteria.
        Possible values: 
        'green' for good landing, 'blue' for touchdown velocity too large
        'orange' for acceleration too large, 'yellow' for projected distance 
        out of bounds
        'purple' for combination of velocity and acceleration too large
        'red' for combination of acceleration and projected distance wrong
        'brown' for combination of velocity and projected distance wrong
        'black' for all wrong
    """
    
    param, dt, ad_function, v_init, alpha_init, method = args
    
    alpha_init_rad = np.deg2rad(alpha_init)
    
    out = reentry(param, dt, ad_function, v_init, alpha_init_rad, method)
    result,_,_,_ = analyse_data(dt, *out, param)
    
    color_map = {(0, 0, 0): 'green',
                 (1, 0, 0): 'blue',
                 (0, 1, 0): 'orange',
                 (0, 0, 1): 'yellow',
                 (1, 1, 0): 'purple',
                 (0, 1, 1): 'red',
                 (1, 0, 1): 'brown',
                 (1, 1, 1): 'black'}
    
    result_tuple = tuple(result)
    color = color_map[result_tuple]
        
    return v_init, alpha_init, color


def many_pairs_mp(num_processes, param, dt, ad_function, v_array, alpha_array, 
                  method):
    """
    Simulates multiple re-entry scenarios in parallel with different initial 
    velocities and angles, and plots the results indicating whether each 
    scenario meets the specified criteria.
    
    - Creates a list of arguments for each combination of initial velocity and 
    angle.
    - Uses multiprocessing to simulate re-entry scenarios in parallel.
    - Collects the results and plots a scatter plot of initial velocities vs 
    initial angles, color-coded based on whether each scenario meets the 
    specified criteria.

    Parameters
    ----------
    num_processes : int
        Number of parallel processes to use.
    param : dict
        Dictionary containing physical parameters required for the simulation, 
        such as mass, cross-sectional area, drag coefficient, lift coefficient, 
        etc.
    dt : float
        Time step used in the re-entry simulation.
    ad_function : function
        Function that computes air density as a function of altitude.
    v_array : numpy array
        Array of initial velocities to test.
    alpha_array : numpy array
        Array of initial angles (in degrees) to test.
    method : function
        Numerical integration method (e.g., Euler, RK2) used to compute the 
        trajectory.
    """
    
    args = [(param, dt, ad_function, v, alpha, method) for v in v_array 
            for alpha in alpha_array]
    
    pool = mp.Pool(processes = num_processes)  
    results = pool.map(many_pairs_parallel, args)
    v_vals, a_vals, colors = zip(*results)
            
    plt.scatter(v_vals, a_vals, c=colors, s=1)
    plt.xlabel('Initial Velocity (m/s)')
    plt.ylabel('Initial Angle (deg)')
    plt.title('Initial Values')
    
    plt.show()
    
    
#%%
def one_pair(param, dt, ad_function, v_init, alpha_init, method):
    """
    Simulates a single re-entry scenario with given initial velocity and angle,
    and plots the results. Prints the outcomes for velocity, acceleration, and 
    projected distance.

    Parameters
    ----------
    param : dict
        Dictionary containing physical parameters required for the simulation, 
        such as mass, cross-sectional area, drag coefficient, lift coefficient, 
        etc.
    dt : float
        Time step used in the re-entry simulation.
    ad_function : function
        Function that computes air density as a function of altitude.
    v_init : float
        Initial velocity for the simulation.
    alpha_init : float
        Initial angle (in radians) for the simulation.
    method : function
        Numerical integration method (e.g., Euler, RK2) used to compute the 
        trajectory.
    """
    
    out = reentry(param, dt, ad_function, v_init, alpha_init, method)
    plot_data(dt, *out)
    result, max_g, last_v, proj_dist = analyse_data(dt, *out, param)
    
    if result[0]:
        print(f"Touchdown Velocity too large: {last_v:.2f} m/s")
    else:
        print(f"Good Touchdown Velocity: {last_v:.2f} m/s")
    if result[1]:
        print(f"Acceleration too large: {max_g:.2f} g")
    else:
        print(f"Good Acceleration: {max_g:.2f} g")
    if result[2]:
        print(f"Projected Distance out of bounds: {proj_dist:.2f} m")
    else: 
        print(f"Good Projected Distance: {proj_dist:.2f} m")
    

#%%
def main():
    
    # Params Dictionary
    paramReentry = {'m_sm': 12000., 'A_sm': 4*np.pi, 'Cd': 1.2, 'Cl': 1., 
                    'A_p': 301., 'Cdp': 1., 'size': 12000, 'h': 1.3*10**5,
                    'h_p': 1000., 'v_p': 100., 'last_v': 25., 'max_g': 15., 
                    'dist_min': 2.5*10**6, 'dist_max': 4.5*10**6, 
                    'newt_tol': 5*10**(-10), 'newt_max_it': 4, \
                    'deriv_dp': 10**(-5)}

    dt = 0.1
    filename = 'airdensity.txt'
    
    # Air Density Function choice. Options: ad_nasa(), ad_cubic_spline(), ad_exp()
    ad_function = ad_nasa(filename, 0) 
    
    # Method choice. Options: backward_euler
    method = backward_euler
    
    if __name__ == "__main__":
        
        # Analysing many pairs of initial values with MP
        v_array = np.arange(7000, 11000, 25)
        alpha_array = np.arange(0, 5, 0.05)
        num_processes = 10
        many_pairs_mp(num_processes, paramReentry, dt, ad_function, v_array, 
                      alpha_array, method)
        
        # Plotting 1 pair of values
        v_init = 9000.
        alpha_init = np.deg2rad(3.)
        one_pair(paramReentry, dt, ad_function, v_init, alpha_init, method)
        
        end_time = time.time()
        print("Runtime:", end_time - start_time, "seconds")
    
   
start_time = time.time()    
main()