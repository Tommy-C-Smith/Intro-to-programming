#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forced Oscillations assignment
This code is designed to model an oscillating system which decays with time.
The user is invited to enter the frequency of the oscillation, the 
value of the constant a1 and the value of the minimum fractional intensity If_min. 
The code will return the number of oscillations and the time taken and plot a graph
of fractional intensity against time.
Created on Sat Oct 21 08:47:22 2023

@author: Tommy curson smith  UID: 11013448
"""

import numpy as np
import matplotlib.pyplot as plt

# constants
A_0 = 2.0  # m^-1

# function definitions
def get_frequency():
    '''
    function that asks user to enter a value of the frequency, 
    validates and returns it.

    Returns
    -------
    frequency : float
    '''
    print('what is the frequency of oscillation in Hz?')
    while True:
        try:
            frequency = float(
                input('please enter a number in the range 1 - 200.  '))
            if not 1 <= frequency <= 200:
                print('this value is outside the range. ')
            else:
                return frequency
        except ValueError:
            print('this is an invalid input please input a number')


def get_a1():
    '''
    function that asks user to enter the value of a1, validates and returns it.

    Returns
    -------
    a1 : float
    '''
    print('what is the value of the constant a1 in m^-1s^-2')
    while True:
        try:
            a1 = float(input('please enter a number in the range 0.1 - 50.  '))
            if not 0.1 <= a1 <= 50:
                print('this value is outside the range.')
            else:
                return a1
        except ValueError:
            print('this is an invalid input, please input a number')


def get_minimum_fractional_intensity():
    '''
    function that asks user to enter the value of If_min, validates and returns it.

    Returns
    -------
    If_min : float
    '''
    print('what is the minimum fractional intensity of the system  for oscillation detection?')
    while True:
        try:
            If_min = float(
                input('please enter a number in the range 0 - 1. 0 is not a valid input however. '))
            if not 0 < If_min <= 1:
                print('this value is outside the range. ')
            else:
                return If_min
        except ValueError:
            print('this is an invalid input please input a number')


def number_oscillations(a_1, a_2):
    '''
    This function defines the fractional intensity of the peaks (If_n) as an expression of the
    integer n. The integer n is then set to 0 and increases by 1 until If_n falls below If_min at
    which point it returns n.

    Parameters
    ----------
    a_1 : float
        a1 constant entered by user.
    a_2 : float
        .angular frequency of oscillation 

    Returns
    -------
    n : integer
        number of oscillations inluding first peak below If_min

    '''
    n = 0
    If_n = ((np.cos(np.pi * n)/(A_0 + a1 * (np.pi * n / a2)**2)) * A_0)**2

    while If_n >= If_min:
        n += 1
        If_n = ((np.cos(np.pi * n)/(A_0 + a1 * (np.pi * n / a2)**2)) * A_0)**2
    return n


# variables/calling functions
frequency = get_frequency()
print('frequency is', frequency, 'Hz')

a1 = get_a1()
print('a1 is', a1, 'm^-1s^-2')
If_min = get_minimum_fractional_intensity()

a2 = 2 * np.pi * frequency

n_osc = number_oscillations(a1, a2) - 1

if If_min == 1 :  #this takes account for extra half period being mistakenly added for 0 oscillations. 
    t_osc = 0
else:    
    t_osc = np.pi * (n_osc + 1/2) / a2

print('the system oscillates', n_osc, 'times before falling below the minimum fractional intensity',
      If_min, '. This oscillation lasts for {0:.3f} seconds.'.format(t_osc))

# plotting a graph (optional)
time_units = 1000
t = np.linspace(0, t_osc + 0.1, time_units)
fractional_intensity = ((np.cos(a2 * t)/(A_0 + a1 * (t)**2)) * A_0)**2

while True:
    graph_option = input('would you like to plot a graph of fractional intensity against time?')
    try:
        if graph_option == 'yes' or graph_option == 'y':
            plt.plot(t, fractional_intensity)
            plt.axhline(y = If_min, color = 'r', linestyle = 'dashed',
                        label = 'minimum fractional intensity')
            plt.axvline(x = t_osc, color = 'g', linestyle = 'dashed', label = 'oscillation time')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Fractional Intensity')
            plt.title('Fractional Intensity vs. Time')
            plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper center')
            plt.grid(True)
            plt.show()
            break
        elif graph_option == 'no' or graph_option == 'n':
            print ('thank you, program has terminated.')
            break 
        else:
            print('please enter a valid response.')
    except:
        print('an error occured, please try again. ')
        
         
            
