#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
    Timing argument calculations
    Author: Lorne Whiteway.
"""


import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
import sys

# r = physical distance between MW and M31
# This evolves according to a differential equation involving G, their total mass M and Lambda.
# For details see arXiv:1308.0970.


# Solve the DE for specified inital conditions and time step. We stop if r becomes negative (and all remaining
# values are set to -1).
# Units:
# r_max (maximum value of r) in Mpc, M in 10^12 solar masses, t_now in Gy, H_0 in km/s/Mpc
# Return vector of r values (in Mpc) at equally spaced time points has length N. It starts at r_max and then falls.
def r_vector(r_max, M, time_step, N, Omega_lambda, H_0):
    assert r_max > 0, "Negative r_max encountered."
    GM = M * 4.50029385227242E-3 # In Mpc^3 Gy^-2
    Lambda_c_squared_over_3 = Omega_lambda * H_0**2 * 1.04566436906929E-06 # In Gy^-2
    r = -1 * np.ones(N) # In Mpc. Default value of -1 denotes 'not calculated (yet)'.
    for i in range(N):
        if i == 0:
            r[i] = r_max
        else:
            prev_r = r[i-1]
            acceleration = (Lambda_c_squared_over_3 * prev_r - GM / prev_r**2)
            
            if i == 1:
                r[i] = prev_r + (time_step**2 / 2.0) * acceleration
            else:
                prev_prev_r = r[i-2]
                r[i] = 2.0 * prev_r - prev_prev_r + time_step**2 * acceleration
        if r[i] < 0.0:
            break
    return r
                
        

# Find a value for the time_step so that r[N-1] = 0.
def solve_for_time_step(r_max, M, N, Omega_lambda, H_0):

    # Initial guess for time_step is intentionally too short; in the while loop we will refine this guess
    # by increasing the time step (slowly at first) until r[N-1] = 0
    
    time_step = 1.0 / float(N) # In Gy
    time_step_epsilon = 0.1 / float(N) # In Gy
    
    while True:
    
        proposed_time_step = time_step + time_step_epsilon
        r = r_vector(r_max, M, proposed_time_step, N, Omega_lambda, H_0)
        
        if r[N-1] < 0.0:
            # We overshot - try with a smaller time step
            time_step_epsilon *= 0.5
        else:
            time_step = proposed_time_step
            time_step_epsilon *= 1.3
            if r[N-1] < 1e-6:
                # Close enough - exit loop
                return time_step



# Units:
# r_max (maximum value of r) in Mpc, M in 10^12 solar masses, t_now in Gy, H_0 in km/s/Mpc
# Return values are r_now in Mpc and v_now in km/s
# r_max is the maximum value of r. N sets the number of time steps between t(r=r_max) and t(r=0).
def r_now_and_v_now(r_max, M, t_now, N, Omega_lambda, H_0):

    # First infer the time step that yields r[N-1]=0.
    time_step = solve_for_time_step(r_max, M, N, Omega_lambda, H_0)
    r = r_vector(r_max, M, time_step, N, Omega_lambda, H_0)

    # What (non-integer) index refers to t_now?
    # Note that we need to subtract N-1 as we have already passed the point of maximum separation.
    # [Note that the time from r=r_max to r=0 (and by symmetry the time from r=0 to r= r_max) is
    # time_step*(N-1), not time_step*N.]
    index_now = t_now/time_step - (N-1)
    int_index_now = int(index_now)
    
    if int_index_now < 1:
        # An alternative solution would be to extend the r vector to negative indices using time symmetry...
        return (r_max, 0.0)
    elif int_index_now >= N-3:
        # An alternative solution would be to extend the r vector to indices greater than N using symmetry...
        return (0.0, -1e4)
    
    # To evaluate (and evaluate the derivative) and the non-grid point 'index_now', we fit a polynomial to nearby points.
    x = np.array(range(int_index_now-1, int_index_now+3))
    quad_fit = np.polyfit(x, r[x], 2)
    r_now = np.polyval(quad_fit, index_now) # In Mpc
    v_now = (np.polyval(np.polyder(quad_fit), index_now) / time_step) # In Mpc/Gy
    
    v_now *= 977.921163963219 # In km/s
    
    return (r_now, v_now)
    


#### Units:
#### r1 in Mpc, M in 10^12 solar masses, t_now in Gy, H_0 in km/s/Mpc
#### return values are r_now in Mpc and v_now in km/s
#### r1 is the value of r at time step 1 (i.e. at t_now/num_time_steps).
###def r_now_and_v_now_old(r1, M, t_now, num_time_points, Omega_lambda, H_0):
###
###    t = np.linspace(0, t_now, num_time_points) # In Gy
###    h = t[1] # In Gy
###    
###    GM = M * 4.50029385227242E-3 # In Mpc^3 Gy^-2
###    Lambda_c_squared_over_3 = Omega_lambda * H_0**2 * 1.04566436906929E-06 # In Gy^-2
###    
###    r = np.zeros(num_time_points) # In Mpc
###    r[0] = 0.0 # Just to be explicit...
###    r[1] = r1
###    for i in range(2, num_time_points):
###        prev_r = r[i-1]
###        prev_prev_r = r[i-2]
###        r[i] = 2 * prev_r - prev_prev_r + h**2 * (Lambda_c_squared_over_3 * prev_r - GM / prev_r**2)
###        
###    r_now = r[num_time_points - 1]
###    v_now = ((r_now - r[num_time_points - 2]) / h) * 977.921163963219 # In km/s
###    
###    return (r_now, v_now)
    
    
def obj_function(x, extra_args):
    # Units for all of these are the same as the interface to r_now_and_v_now
    t_now = extra_args[0]
    N = int(extra_args[1])
    Omega_lambda = extra_args[2]
    H_0 = extra_args[3]
    target_r_now = extra_args[4]
    target_v_now = extra_args[5]
    
    r_max = x[0]
    M = x[1]
    
    (r_now, v_now) = r_now_and_v_now(r_max, M, t_now, N, Omega_lambda, H_0)
    
    r_err = r_now - target_r_now
    v_err = v_now - target_v_now
    
    #print(r_max, M, r_err, v_err)
    
    return np.array([r_err, v_err])
    
    
def show_plot_of_solution(r, time_step, N, t_now):

    t = (N + np.arange(N)) * time_step
    plt.scatter(t, r)
    plt.show()
    
    
    
    
def solve_for_r_max_and_M(r_now, v_now, t_now, N, Omega_lambda, H_0, guess_r_max, guess_M):

    fun = obj_function
    x0 = np.array([guess_r_max, guess_M])
    args = np.array([t_now, N, Omega_lambda, H_0, r_now, v_now])

    root_ret = sp.root(fun = fun, x0 = x0, args = args, method = 'broyden1')
    
    if not root_ret.success:
        raise AssertionError(root_ret.message)
        
    r_max = root_ret.x[0]
    M = root_ret.x[1]
    
    plot_solution = False
    if plot_solution:
        time_step = solve_for_time_step(r_max, M, N, Omega_lambda, H_0)
        r = r_vector(r_max, M, time_step, N, Omega_lambda, H_0)
        show_plot_of_solution(r, time_step, N, t_now)
    
    return(r_max, M)




def regression_test():
    target_r_now = 0.784 # Mpc
    target_v_now = -130.0 # km/s
    t_now = 13.81 # Gy
    N = 5000
    Omega_lambda = 0.69
    H_0 = 67.4 # km/s/Mpc
    guess_r_max = 1.1 # Mpc
    guess_M = 5.95 # 10^12 solar masses
    
    expected_r_max = 1.1002669906154712
    expected_M = 5.947007980090099
    
    (r_max, M) = solve_for_r_max_and_M(target_r_now, target_v_now, t_now, N, Omega_lambda, H_0, guess_r_max, guess_M)
    
    r_max_OK = abs(r_max - expected_r_max) < 1e-3
    M_OK = abs(M - expected_M) < 1e-3
    
    print(r_max, M)

    print("Regression test " + ("passed" if r_max_OK and M_OK else "FAILED") + ".")
    
    

    
    

if __name__ == '__main__':


    regression_test()
    sys.exit()

    
    target_r_now = 0.784 # Mpc
    target_v_now = -130.0 # km/s
    t_now = 13.81 # Gy
    N = 5000
    Omega_lambda = 0.69
    H_0 = 67.4 # km/s/Mpc
    guess_r_max = 1.1 # Mpc
    guess_M = 6.0 # 10^12 solar masses
    
    
    for t_now in np.linspace(13.56, 14.06, 11):
        (r_max, M) = solve_for_r_max_and_M(target_r_now, target_v_now, t_now, N, Omega_lambda, H_0, guess_r_max, guess_M)
        print(target_r_now, target_v_now, t_now, Omega_lambda, H_0, r_max, M)
    
    
    
    
