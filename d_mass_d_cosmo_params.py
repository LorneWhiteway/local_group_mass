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
import math

# r = physical distance between MW and M31
# This evolves according to a differential equation involving G, their total mass M and Lambda.
# For details see arXiv:1308.0970.


# Solve the DE for specified inital conditions and time step. We stop if r becomes negative (and all remaining
# values are set to -1).
# Units:
# r_max (maximum value of r) in Mpc, M in 10^12 solar masses, t_now in Gy, H_0 in km/s/Mpc
# Return vector of r values (in Mpc) at equally spaced time points has length N. It starts at r_max and then falls.
def r_vector(r_max, M, time_step, N, Omega_lambda, H_0):

    # TODO - use these constants instead of magic numbers.
    SECONDS_PER_GIGAYEAR = 3.1556736E+16
    METRES_PER_MEGAPARSEC = 3.0856776E+22
    #INVERSE_SECONDS_PER_KM_OVER_SECOND_OVER_MEGAPARSEC = 1.0E3 / METRES_PER_MEGAPARSEC


    GM = M * 4.50029385227242E-3 # In Mpc^3 Gy^-2
    Lambda_c_squared_over_3 = Omega_lambda * H_0**2 * 1.04566436906929E-06 # In Gy^-2
    time_step_squared = time_step**2 # In Gy^2
    r = -1 * np.ones(N) # In Mpc. Default value of -1 denotes 'not calculated (yet)'.
    for i in range(N):
        if i == 0:
            r[i] = r_max
        else:
            prev_r = r[i-1]
            acceleration = (Lambda_c_squared_over_3 * prev_r - GM / prev_r**2)
            
            # Positive acceleration is disastrous as we never converge to r = 0.
            assert acceleration <= 0.0, "Positive acceleration encountered"
            
            if i == 1:
                # Here we are using dr_dt[0]=0 (as r=r_max at that point).
                r[i] = prev_r + 0.5 * time_step_squared * acceleration
            else:
                prev_prev_r = r[i-2]
                r[i] = 2.0 * prev_r - prev_prev_r + time_step_squared * acceleration
        
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
    
    
# x should contain ln(r_max/1Mpc) and ln(M/(10^12 solar masses)).
def obj_function(x, extra_args):
    # Units for all of these are the same as the interface to r_now_and_v_now
    t_now = extra_args[0]
    N = int(extra_args[1])
    Omega_lambda = extra_args[2]
    H_0 = extra_args[3]
    target_r_now = extra_args[4]
    target_v_now = extra_args[5]
    
    r_max = math.exp(x[0])
    M = math.exp(x[1])
    #print(r_max, M, x)
    
    (r_now, v_now) = r_now_and_v_now(r_max, M, t_now, N, Omega_lambda, H_0)
    
    r_err = r_now - target_r_now
    v_err = v_now - target_v_now

    return np.array([r_err, v_err])
    
    
def show_plot_of_solution(r, time_step, N, t_now):

    t = (N + np.arange(N)) * time_step
    plt.scatter(t, r)
    plt.show()
    
    
    
    
def inferred_r_max_and_M(r_now, v_now, t_now, N, Omega_lambda, H_0, guess_r_max, guess_M, print_header):

    fun = obj_function
    
    # In our interaction with the root finder we use log_r_max and log_M
    # (as we want to ensure r_max and M are both positive and the root
    # finder doesn't allow constraints to be specified).
    x0 = np.log(np.array([guess_r_max, guess_M]))
    args = np.array([t_now, N, Omega_lambda, H_0, r_now, v_now])

    root_ret = sp.root(fun = fun, x0 = x0, args = args, method = 'hybr')
    
    if not root_ret.success:
        raise AssertionError(root_ret.message)
        
    r_max = math.exp(root_ret.x[0])
    M = math.exp(root_ret.x[1])
    
    plot_solution = False
    if plot_solution:
        time_step = solve_for_time_step(r_max, M, N, Omega_lambda, H_0)
        r = r_vector(r_max, M, time_step, N, Omega_lambda, H_0)
        show_plot_of_solution(r, time_step, N, t_now)
        
    if print_header:
        print("r_now", "v_now", "t_now", "N", "Omega_lambda", "H_0", "r_max", "M")
    print(r_now, v_now, t_now, N, Omega_lambda, H_0, r_max, M)
    
    return(r_max, M)


def regression_test():
    
    print("Running regression test...")
    
    target_r_now = 0.784 # Mpc
    target_v_now = -130.0 # km/s
    t_now = 13.81 # Gy
    N = 5000
    Omega_lambda = 0.69
    H_0 = 67.4 # km/s/Mpc
    guess_r_max = 1.1 # Mpc
    guess_M = 5.95 # 10^12 solar masses
    
    expected_r_max = 1.1002669906154712 # Gy
    expected_M = 5.947007980090099 # 10^12 solar masses
    
    (r_max, M) = inferred_r_max_and_M(target_r_now, target_v_now, t_now, N, Omega_lambda, H_0, guess_r_max, guess_M, True)
    
    r_max_OK = abs(r_max - expected_r_max) < 1e-3
    M_OK = abs(M - expected_M) < 1e-3

    print("Regression test " + ("passed" if r_max_OK and M_OK else "FAILED") + ".")
    
    
# Units:
# r_now in Mpc, v_now in km/s, t_now in Gy, H_0 in km/s/Mpc, guess_r_max in Mpc, guess_M in 10^12 solar masses.
# The same unis are used in the return value.
def M_and_derivatives(r_now, v_now, t_now, N, Omega_lambda, H_0, guess_r_max, guess_M):
    # Base case
    (r_max, M) = inferred_r_max_and_M(r_now, v_now, t_now, N, Omega_lambda, H_0, guess_r_max, guess_M, True)
    
    # Use the base case results as the intial guess when calculating derivatives.
    #print(guess_r_max, r_max, guess_M, M)
    guess_r_max = r_max
    guess_M = M
    
    r_now_delta = 0.001 # Mpc
    (_, M_up) = inferred_r_max_and_M(r_now + r_now_delta, v_now, t_now, N, Omega_lambda, H_0, guess_r_max, guess_M, False)
    (_, M_dn) = inferred_r_max_and_M(r_now - r_now_delta, v_now, t_now, N, Omega_lambda, H_0, guess_r_max, guess_M, False)
    dM_d_r_now = (M_up - M_dn) / (2.0 * r_now_delta) # 10^12 solar masses/Mpc
    
    v_now_delta = 1.0 # km/s
    (_, M_up) = inferred_r_max_and_M(r_now, v_now + v_now_delta, t_now, N, Omega_lambda, H_0, guess_r_max, guess_M, False)
    (_, M_dn) = inferred_r_max_and_M(r_now, v_now - v_now_delta, t_now, N, Omega_lambda, H_0, guess_r_max, guess_M, False)
    dM_d_v_now = (M_up - M_dn) / (2.0 * v_now_delta) # 10^12 solar masses/(km/s)
    
    t_now_delta = 0.01 # Gy
    (_, M_up) = inferred_r_max_and_M(r_now, v_now, t_now + t_now_delta, N, Omega_lambda, H_0, guess_r_max, guess_M, False)
    (_, M_dn) = inferred_r_max_and_M(r_now, v_now, t_now - t_now_delta, N, Omega_lambda, H_0, guess_r_max, guess_M, False)
    dM_d_t_now = (M_up - M_dn) / (2.0 * t_now_delta) # 10^12 solar masses/Gy
    
    Omega_lambda_delta = 0.01 # unitless
    (_, M_up) = inferred_r_max_and_M(r_now, v_now, t_now, N, Omega_lambda + Omega_lambda_delta, H_0, guess_r_max, guess_M, False)
    (_, M_dn) = inferred_r_max_and_M(r_now, v_now, t_now, N, Omega_lambda - Omega_lambda_delta, H_0, guess_r_max, guess_M, False)
    dM_d_Omega_lambda = (M_up - M_dn) / (2.0 * Omega_lambda_delta) # 10^12 solar masses
    
    H_0_delta = 1.0 # km/s/Mpc
    (_, M_up) = inferred_r_max_and_M(r_now, v_now, t_now, N, Omega_lambda, H_0 + H_0_delta, guess_r_max, guess_M, False)
    (_, M_dn) = inferred_r_max_and_M(r_now, v_now, t_now, N, Omega_lambda, H_0 - H_0_delta, guess_r_max, guess_M, False)
    dM_d_H_0 = (M_up - M_dn) / (2.0 * H_0_delta) # 10^12 solar masses / (km/s/Mpc)
    
    return(M, dM_d_r_now, dM_d_v_now, dM_d_t_now, dM_d_Omega_lambda, dM_d_H_0)



def error_analysis():


    r_now = 0.784 # Mpc
    v_now = -130.0 # km/s
    t_now = 13.81 # Gy
    N = 5000
    Omega_lambda = 0.69 # unitless
    H_0 = 67.4 # km/s/Mpc
    guess_r_max = 1.1 # Mpc
    guess_M = 5.95 # 10^12 solar masses
    
    (M, dM_d_r_now, dM_d_v_now, dM_d_t_now, dM_d_Omega_lambda, dM_d_H_0) = M_and_derivatives(r_now, v_now, t_now, N, Omega_lambda, H_0, guess_r_max, guess_M)
    
    print("\n===============\n")
    
    print("Sensitivity of M to Omega_lambda = {} 10^12 solar masses".format(dM_d_Omega_lambda))
    
    print("Sensitivity of M to H0 (via Lambda, with Omega_lambda fixed) = {} 10^12 solar masses / (km/s/Mpc)".format(dM_d_H_0))
    
    d_t_now_d_H0 = -t_now/H_0 # In Gy /(km/s/Mpc)
    dM_d_H_0_via_t_now = dM_d_t_now * d_t_now_d_H0 # In 10^12 solar masses / (km/s/Mpc)
    print("Sensitivity of M to H0 (via t_now) = {} 10^12 solar masses / (km/s/Mpc)".format(dM_d_H_0_via_t_now))
    
    print("\n===============\n")
    
    error_in_omega_lambda = 0.01
    error_in_M_from_omega_lambda = error_in_omega_lambda * dM_d_Omega_lambda

    error_in_H_0 = 0.5 # km/s/Mpc
    error_in_M_from_H0 = (dM_d_H_0 + dM_d_H_0_via_t_now) * error_in_H_0
    
    print("Error in M from Omega_lambda = {} 10^12 solar masses (assuming error in Omega_lambda = {})".format(error_in_M_from_omega_lambda, error_in_omega_lambda))

    print("Error in M from H0 (both sources) = {} 10^12 solar masses (assuming error in H0 = {} km/s/Mpc)".format(error_in_M_from_H0, error_in_H_0))
    
    print("\n===============\n")
    

if __name__ == '__main__':

    do_regression_test = ((len(sys.argv) > 1 and sys.argv[1] == "test"))

    if do_regression_test:
        regression_test()
    else:
        error_analysis()
        
        
        
    
    
