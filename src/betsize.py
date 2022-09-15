import itertools as it
import numpy as np
from scipy.optimize import minimize

def getOptimalK(num_bets, win_return, loss_return, win_probability):
    '''
    This function finds the optimal bet size K for a number of identical and simultaneous bets.
    :param num_bets: The integer number of simultaneous bets
    :param win_return: The expected return of a single bet if it wins
    :param loss_return: The expected loss of a single bet if it loses
    :param win_probability: The single bet probability that a bet will win
    :return: [K, GEV] K is the optimal betsize of the combined bets... 
                      GEV is the geometric expected value of all the bets using bet size K
    '''
    # Prevent dimensionality explosion
    num_bets = min(num_bets, 20)
    
    # Get the list of outcome combinations of N bets
    combo_prob = list(it.product([win_probability, 1-win_probability], repeat=num_bets))
    combo_ret = list(it.product([win_return, loss_return], repeat=num_bets))
    
    # compute joint probabilties of outcomes
    P = []
    for combo in combo_prob:
        product = 1
        for i in combo: 
            product *= i
        P.append(product)
    P = np.array(P)
    
    #compute joint returns of outcomes
    r =[]
    for combo in combo_ret:
        r.append(sum(combo)/num_bets)
    r = np.array(r)
    
    # Find K that minimizes the GEV
    x0 = np.array([0.5])
    result = minimize(GEV, x0, method='nelder-mead', args=(P, r), options={'xatol': 1e-8, 'disp': False})
    
    return [result.final_simplex[0][0],-result.final_simplex[1][0]]

def GEV(K,_prob,_ret):
    '''This function calculates the geometric expected value given the ending wealths of all outcomes
    :param _prob: An array of all the probabilities of each outcome
    :param _ret: An array of the return of each outcome
    :param K: The betsize as a percentage of total portfolio alocated to all bets
    :return: It returns the the GEV*-1. This is so a minimize algortithm can be used to find the optimal K
    '''
    summation = _prob*np.log(1+K*_ret)
    gev = np.exp(sum(summation))
    return -gev
