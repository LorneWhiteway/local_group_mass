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

SECONDS_PER_GIGAYEAR = 3.1556736E+16
METRES_PER_MEGAPARSEC = 3.0856776E+22

GRAVITATIONAL_CONSTANT = 6.6743E-11 # m^3 kg^-1 s^-2
SOLAR_MASS = 1.98847E30 # kg



# Solve the DE for specified inital conditions and time step. We stop if r becomes negative (and all remaining
# values are set to -1).
# Units:
# r_max (maximum value of r) in Mpc, M in 10^12 solar masses, t_now in Gy, H_0 in km/s/Mpc
# Return vector of r values (in Mpc) at equally spaced time points has length N. It starts at r_max and then falls.
def r_vector(r_max, M, time_step, N, Omega_lambda, H_0):

    GM = GRAVITATIONAL_CONSTANT * M * (SOLAR_MASS * 1.0e12 * SECONDS_PER_GIGAYEAR**2 * METRES_PER_MEGAPARSEC**(-3)) # In Mpc^3 Gy^-2
    
    Lambda_c_squared_over_3 = Omega_lambda * (H_0**2 * SECONDS_PER_GIGAYEAR**2 * METRES_PER_MEGAPARSEC**(-2) * 1.0e6) # In Gy^-2
    
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
                # This is identical to the leapfrog algorithm. We equate
                # numerical and analytic expressions for the acceleration at the 
                # previous step, and solve for r[i].
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
    
    # In order to evaluate (and evaluate the derivative) at the non-grid point 'index_now',
    # we fit a polynomial to nearby points.
    x = np.array(range(int_index_now-1, int_index_now+3))
    quad_fit = np.polyfit(x, r[x], 2)
    r_now = np.polyval(quad_fit, index_now) # In Mpc
    v_now = (np.polyval(np.polyder(quad_fit), index_now) / time_step) # In Mpc/Gy

    v_now *= (1.0e-3 * METRES_PER_MEGAPARSEC / SECONDS_PER_GIGAYEAR) # In km/s
    
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
    
    expected_r_max = 1.100301473080796 # Gy
    expected_M = 5.950378614313688 # 10^12 solar masses
    
    (r_max, M) = inferred_r_max_and_M(target_r_now, target_v_now, t_now, N, Omega_lambda, H_0, guess_r_max, guess_M, True)
    
    r_max_OK = abs(r_max - expected_r_max) < 1e-3
    M_OK = abs(M - expected_M) < 1e-3

    print("Regression test " + ("passed" if r_max_OK and M_OK else "FAILED") + ".")
    
    
# Units:
# r_now in Mpc, v_now in km/s, t_now in Gy, H_0 in km/s/Mpc, guess_r_max in Mpc, guess_M in 10^12 solar masses.
# The same unis are used in the return value.
def M_and_derivatives_old_parameterization(r_now, v_now, t_now, N, Omega_lambda, H_0, guess_r_max, guess_M):
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



def combine_in_quadrature(std1, std2):
    return math.sqrt(std1**2 + std2**2)


# Returns abs(d_mass_d_parameter * parameter_uncertainty) in 10^12 solar masses
def print_one_uncertainty_item(parameter, parameter_unit, d_mass_d_parameter, parameter_uncertainty):
    print("{}: sensitivity = {:4.4f} 10^12 solar masses{}; parameter uncertainty = {:4.4f}{}; mass uncertainty = {:4.4f} 10^12 solar masses".format(parameter, d_mass_d_parameter, ("/" if len(parameter_unit) > 0 else "") + parameter_unit, parameter_uncertainty, (" " if len(parameter_unit) > 0 else "") + parameter_unit, abs(d_mass_d_parameter * parameter_uncertainty)))
    
    return abs(d_mass_d_parameter * parameter_uncertainty)




def uncertainty_analysis_1():


    r_now = 0.784 # Mpc
    v_now = -130.0 # km/s
    t_now = 13.81 # Gy
    N = 1000
    Omega_lambda = 0.69 # unitless
    H_0 = 67.4 # km/s/Mpc
    guess_r_max = 1.1 # Mpc
    guess_M = 5.95 # 10^12 solar masses
    
    (M, dM_d_r_now, dM_d_v_now, dM_d_t_now, dM_d_Omega_lambda, dM_d_H_0) = M_and_derivatives_old_parameterization(r_now, v_now, t_now, N, Omega_lambda, H_0, guess_r_max, guess_M)
    
    uncertainty_in_omega_lambda = 0.006
    uncertainty_in_H_0 = 0.4 # km/s/Mpc
    uncertainty_in_t_now = 0.024 # Gy
    
    print("\n")
    
    print_one_uncertainty_item("Omega_lambda", "", dM_d_Omega_lambda, uncertainty_in_omega_lambda)
    print_one_uncertainty_item("H_0", "km/s/Mpc", dM_d_H_0, uncertainty_in_H_0)
    print_one_uncertainty_item("t_now", "Gy", dM_d_t_now, uncertainty_in_t_now)
    
    
# Units:
# r_now in Mpc, v_now in km/s, H_0 in km/s/Mpc, guess_r_max in Mpc, guess_M in 10^12 solar masses.
# The same unis are used in the return value.
def M_new_parameterization(r_now, v_now, N, Omega_lambda, H_0, guess_r_max, guess_M, printHeader):
    t_now = age_of_universe(Omega_lambda, H_0) # Gy
    (_, M) = inferred_r_max_and_M(r_now, v_now, t_now, N, Omega_lambda, H_0, guess_r_max, guess_M, printHeader)
    return M
    
    
def M_and_derivatives_new_parameterization(r_now, v_now, N, Omega_lambda, H_0, guess_r_max, guess_M):

    M = M_new_parameterization(r_now, v_now, N, Omega_lambda, H_0, guess_r_max, guess_M, True)
    
    Omega_lambda_delta = 0.01 # unitless
    M_up = M_new_parameterization(r_now, v_now, N, Omega_lambda + Omega_lambda_delta, H_0, guess_r_max, guess_M, False)
    M_dn = M_new_parameterization(r_now, v_now, N, Omega_lambda - Omega_lambda_delta, H_0, guess_r_max, guess_M, False)
    dM_d_Omega_lambda = (M_up - M_dn) / (2.0 * Omega_lambda_delta) # 10^12 solar masses
    
    H_0_delta = 1.0 # km/s/Mpc
    M_up = M_new_parameterization(r_now, v_now, N, Omega_lambda, H_0 + H_0_delta, guess_r_max, guess_M, False)
    M_dn = M_new_parameterization(r_now, v_now, N, Omega_lambda, H_0 - H_0_delta, guess_r_max, guess_M, False)
    dM_d_H_0 = (M_up - M_dn) / (2.0 * H_0_delta) # 10^12 solar masses / (km/s/Mpc)
    
    return(M, dM_d_Omega_lambda, dM_d_H_0)

    
    
    
def uncertainty_analysis_2():    
    
    r_now = 0.784 # Mpc
    v_now = -130.0 # km/s
    N = 1000
    Omega_lambda = 0.69 # unitless
    H_0 = 67.0 # km/s/Mpc
    guess_r_max = 1.1 # Mpc
    guess_M = 5.95 # 10^12 solar masses
    
    (M, dM_d_Omega_lambda, dM_d_H_0) = M_and_derivatives_new_parameterization(r_now, v_now, N, Omega_lambda, H_0, guess_r_max, guess_M)
    
    uncertainty_from_posterior_width = 2.3 # 10^12 solar masses
    
    for uncertainty_in_H_0 in [0.4, 4.0]: # km/s/Mpc

        print("============")
        
        uncertainty_in_omega_lambda = 0.006
        mass_uncertainty_1 = print_one_uncertainty_item("Omega_lambda", "", dM_d_Omega_lambda, uncertainty_in_omega_lambda)
        mass_uncertainty_2 = print_one_uncertainty_item("H_0", "km/s/Mpc", dM_d_H_0, uncertainty_in_H_0)
        
        mass_uncertainty_1_2 = combine_in_quadrature(mass_uncertainty_1, mass_uncertainty_2)
        print("Uncertainty from H_0 and Omega_lambda (combined in quadrature): {} 10^12 solar masses".format(mass_uncertainty_1_2))
        
        total_mass_uncertainty = combine_in_quadrature(uncertainty_from_posterior_width, mass_uncertainty_1_2)
        print("Total Uncertainty (combined in quadrature with uncertainty of {} 10^12 solar masses from posterior width): {} 10^12 solar masses".format(uncertainty_from_posterior_width, total_mass_uncertainty))
    
    
    print("============")
    
    
    

# H_0 in km/s/Mpc; return value in Gy
# Equation 70 in https://www.uni-ulm.de/fileadmin/website_uni_ulm/nawi.inst.260/paper/08/tp08-7.pdf
# Ignores radiation.
def age_of_universe(Omega_lambda, H_0):

    Omega_m = 1.0 - Omega_lambda

    factor = (2.0/3.0) * np.log(np.sqrt(Omega_lambda/Omega_m) + np.sqrt(1.0/Omega_m)) / np.sqrt(Omega_lambda)
    
    return factor / (H_0 * 1.0e3 * SECONDS_PER_GIGAYEAR / METRES_PER_MEGAPARSEC)
    
    
# See https://arxiv.org/abs/1903.10849
def comparison_with_michaels_work():
    
    r_now = 0.77 # Mpc
    v_now = -109.4 # km/s
    
    N = 1000
    Omega_lambda = 0.7 # unitless
    #H_0 = 67 # km/s/Mpc
    
    guess_r_max = 1.03 # Mpc
    guess_M = 4.7 # 10^12 solar masses
    
    printHeader = True
    
    for H_0 in np.linspace(60.0, 80.0, 21):
        t_now = age_of_universe(Omega_lambda, H_0) # Gy
        
        (r_max, M) = inferred_r_max_and_M(r_now, v_now, t_now, N, Omega_lambda, H_0, guess_r_max, guess_M, printHeader)
        
        # Use the previous results as the starting point for the next iteration.
        guess_r_max = r_max
        guess_M = M
        printHeader = False
        
def H0_sensitivity_breakdown():

    r_now = 0.784 # Mpc
    v_now = -130.0 # km/s
    N = 1000
    Omega_lambda = 0.69 # unitless
    H_0 = 67.0 # km/s/Mpc
    guess_r_max = 1.1 # Mpc
    guess_M = 5.95 # 10^12 solar masses
    
    t_now = age_of_universe(Omega_lambda, H_0)
    (_, m1) = inferred_r_max_and_M(r_now, v_now, t_now, N, Omega_lambda, H_0, guess_r_max, guess_M, True)
    H_0_prime = H_0+4
    (_, m2) = inferred_r_max_and_M(r_now, v_now, t_now, N, Omega_lambda, H_0_prime, guess_r_max, guess_M, False)
    print(m2-m1)
    t_now_prime = t_now * (H_0 / (H_0+4))
    (_, m3) = inferred_r_max_and_M(r_now, v_now, t_now_prime, N, Omega_lambda, H_0, guess_r_max, guess_M, False)
    print(m3-m1)
    (_, m4) = inferred_r_max_and_M(r_now, v_now, t_now_prime, N, Omega_lambda, H_0_prime, guess_r_max, guess_M, False)
    print(m4-m1)
    
    

if __name__ == '__main__':

    do_regression_test = ((len(sys.argv) > 1 and sys.argv[1] == "test"))

    if do_regression_test:
        regression_test()
    else:
        #uncertainty_analysis_1()
        #comparison_with_michaels_work()
        uncertainty_analysis_2()
        H0_sensitivity_breakdown()
        
        
        
        
    
    
