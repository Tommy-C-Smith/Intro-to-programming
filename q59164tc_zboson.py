#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final assignment: Z0 Boson.
This code extracts the data from two seperate files and uses it to find the mass,
width and lifetime of a Z0 boson in a e-e+ -> Z0 -> e-e+ process. It does this by
fitting data of the boson's cross-section at different centre-of-mass energy values
to a known relation where the two unknown parameters to be optimised are the mass and
width. By minimising the chi squared fit to this relation, these parameters are optimised.
An uncertainty is obtained on these parameters associated with the distance between their
maximum and minimum values on a contour where the chi squared value is equal to the
minimised chi squared value plus one. these contours as well as the data fit are plotted
and obtained as figures.

@author: tommycursonsmith

"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as pc
from scipy.optimize import minimize
from matplotlib import gridspec

FILE_1 = Path("z_boson_data_1.csv")
FILE_2 = Path("Z_boson_data_2.csv")

def read_and_combine_data():
    '''
    This function reads in experimental data from both files and combines them
    into a 2D array whilst ignoring any invalid non number data points.

    Returns
    -------
    data : array
        Combined data from both files.
    '''
    if not (FILE_1.exists() and FILE_2.exists()):
        print("One or both files do not exist.")
        return None
    data_1 = np.genfromtxt(FILE_1, comments='%', delimiter=',')
    data_2 = np.genfromtxt(FILE_2, comments='%', delimiter=',')
    combined_data = np.vstack((data_1, data_2))
    invalid_points = np.isnan(combined_data).any(axis=1)
    data = np.delete(combined_data,invalid_points,axis=0)
    if len(data) == 0:
        print("No valid data available.")
        return None
    return data


def remove_outliers(raw_data):
    '''
    Removes data points associated with outliers in the cross section. Here
    negative values as well as values more than three standard deviations away
    from the mean are removed.

    Parameters
    ----------
    raw_data : array
        Data array from which outliers will be removed.

    Returns
    -------
    filtered_data : array
        Data array with rows containing outliers in the specified column removed.
    '''
    if raw_data is None:
        return None
    cross_section_data = raw_data[:,1]
    mean_value = np.mean(cross_section_data)
    standard_deviation = np.std(cross_section_data)
    outlier_condition = np.where((cross_section_data < 0)
                         | (cross_section_data > mean_value + 3 * standard_deviation))
    filtered_data = np.delete(raw_data,outlier_condition,axis=0)
    return filtered_data


def find_cross_section(mass_z, width_z, energy):
    '''
    this defines the cross section as a function of energy. The two coefficients
    are the parameters that are to be optimized.

    Parameters
    ----------
    mass_z : parameter
        mass of boson, to be optimised.
    width_z : parameter
        width of boson, to be optimised.
    energy : array
        array of energy values.

    Returns
    -------
    cross_section : array
        array of cross section values.
    '''
    cross_section = ((12 * np.pi / (mass_z**2))
                     * (energy**2 / ((energy**2 - mass_z**2)**2 + mass_z**2 * width_z**2))
                     * (0.08391)**2 * (0.3894e6))
    return cross_section


def find_chi_squared(mass_z, width_z, filtered_data, energy):
    '''
    this function calculates the chi-squared value using observed data and the
    predicted relation, ignoring terms where σ_error is zero which cannot be computed.

    Parameters
    ----------
    mass_z : parameter
        Mass of the Z boson in GeV/c^2.
    width_z : parameter
        Width of the Z boson in GeV.
    filtered_data : array
        Filtered experimental data.
    energy : array-like
        Array of energy values.

    Returns
    -------
    chi_squared : float
        Calculated chi-squared value.

    '''
    if filtered_data is None:
        return None
    cross_section_data = filtered_data[:, 1]
    error_data = filtered_data[:, 2]
    predicted_cross_section = find_cross_section(mass_z, width_z, energy)

    #we must ignore data points with zero error as we cant divide by zero
    non_zero_errors = np.where(error_data != 0)[0]
    chi_squared = np.sum(((cross_section_data[non_zero_errors]
                           - predicted_cross_section[non_zero_errors])
                          / error_data[non_zero_errors])**2)
    return chi_squared


def minimise_chi_squared(filtered_data, energy):
    '''
    This function minimises the chi-squared value by varying the two parameters
    M_Z and W_Z until the chi-squared value is minimised. The corresponding parameters
    are optimised.

    Parameters
    ----------
    filtered_data : array
        Filtered data array.
    energy : array
        array of energy values.

    Returns
    -------
    minimised_chi_squared : TYPE
        Result of the minimization process.
    '''
    if filtered_data is None:
        return None
    initial_guess = [90, 3] # these initial guesses are based on previous studies
    func_to_minimize = lambda params: find_chi_squared(params[0], params[1], filtered_data, energy)
    minimised_chi_squared = minimize(func_to_minimize, initial_guess, method='Nelder-Mead')
    return minimised_chi_squared


def find_lifetime(optimized_width):
    '''
    calculates the lifetime of the boson.

    Parameters
    ----------
    optimized_width : float
        width associated with minimised chi_squard value

    Returns
    -------
    lifetime : TYPE
        DESCRIPTION.

    '''
    lifetime = pc.hbar/(optimized_width * pc.electron_volt * 10**9)
    return lifetime


def find_reduced_chi_squared(energy, chi_min):
    '''
    Calculates reduced chi-squared value.

    Parameters
    ----------
    energy : array-like
        Array of energy values.
    chi_min : OptimizeResult
        Result of the minimization process.

    Returns
    -------
    reduced_chi : float
        Reduced chi-squared value.
    '''
    chi_squared_min = chi_min.fun
    reduced_chi = chi_squared_min / (len(energy) - 2)
    return reduced_chi

def uncertainty_contour_plot(filtered_data, energy, chi_min):
    '''
    this function converts a range of mass and width values into mesh grids and
    generates a plot of contours of constant chi squared values over these ranges.
    it also calculates the uncertainty on these parameters which is half the distance
    between their maximum and minimum values on the contour where chi-squared is
    minimised chi squared plus one.

    Parameters
    ----------
    filtered_data : array
        Filtered experimental data.
    energy : array-like
        Array of energy values.
    chi_min : OptimizeResult
        Result of the minimization process.

    Returns
    -------
    mass_uncertainty : float
        Calculated uncertainty for mass.
    width_uncertainty : float
        Calculated uncertainty for width.
    '''
    # use mesh grids to cretae a contour-plot of chi-squared values
    mass_range = np.linspace(chi_min.x[0]-0.05,chi_min.x[0]+0.05 , 100)
    width_range = np.linspace(chi_min.x[1]-0.05,chi_min.x[1]+0.05, 100)
    mass, width = np.meshgrid(mass_range, width_range)

    chi_squared_values = np.zeros_like(mass)
    for i in range(len(mass_range)):
        for j in range(len(width_range)):
            chi_squared_values[i, j] = find_chi_squared(mass[i, j], width[i, j],
                                                        filtered_data, energy)

    fig = plt.figure(figsize=(6,6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    # Add the main contour plot
    ax1 = fig.add_subplot(gs[0])
    contours = ax1.contour(mass, width, chi_squared_values,
                          levels=[chi_min.fun + 1, chi_min.fun + 2.3, chi_min.fun + 5.99])
    ax1.set_xlabel('Mass ($M_z$) - $GeV/c^2$')
    ax1.set_ylabel('Width ($\Gamma_z$) - GeV')
    ax1.set_title('Contours of Chi-squared Values')
    plt.colorbar(contours, ax=ax1, label='Chi-squared')
    ax1.scatter(chi_min.x[0], chi_min.x[1], color='red', label='Minimized Chi-squared', zorder=5)
    ax1.legend()

    ax2 = fig.add_subplot(gs[1])
    ax2.set_axis_off()
    proxy = [plt.Line2D([0], [0], linestyle='-',linewidth = 5,
                        color='purple', label='$χ^2_{min}$ + 1'),
             plt.Line2D([0], [0], linestyle='-',linewidth = 5,
                        color='blue', label='$χ^2_{min}$ + 2.3'),
             plt.Line2D([0], [0], linestyle='-',linewidth = 5,
                        color='yellow', label='$χ^2_{min}$ + 5.99')]
    ax2.legend(handles=proxy, loc='upper center')
    plt.subplots_adjust(hspace=0.1)

    #here we are finding the uncertainty using the contour where chi-squared is its minimum+1
    uncertainty_contour = chi_min.fun + 1
    contour = ax1.contour(mass, width, chi_squared_values, levels=[uncertainty_contour])
    uncertainty_contour_path = contour.collections[0].get_paths()

    max_mass, min_mass = None, None
    max_width, min_width = None, None

    for path in uncertainty_contour_path:
        vertices = path.vertices
        mass_values = vertices[:, 0]
        width_values = vertices[:, 1]

        if max_mass is None or np.max(mass_values) > max_mass:
            max_mass = np.max(mass_values)
        if min_mass is None or np.min(mass_values) < min_mass:
            min_mass = np.min(mass_values)
        if max_width is None or np.max(width_values) > max_width:
            max_width = np.max(width_values)
        if min_width is None or np.min(width_values) < min_width:
            min_width = np.min(width_values)
    #here we plot where these uncertainties are defined
    ax1.axvline(x = min_mass, color = 'g', linestyle = 'dashed')
    ax1.axvline(x = max_mass, color = 'g', linestyle = 'dashed')
    ax1.axhline(y = max_width, color = 'darkred', linestyle = 'dashed')
    ax1.axhline(y = min_width, color = 'darkred', linestyle = 'dashed')
    ax1.text(min_mass + 0.001, 2.462, '$M_z - \Delta M$', color='g')
    ax1.text(max_mass + 0.001, 2.462, '$M_z + \Delta M$', color='g')
    ax1.text(91.13, max_width + 0.003, '$\Gamma_z + \Delta \Gamma$', color='darkred')
    ax1.text(91.13, min_width + 0.003, '$\Gamma_z - \Delta \Gamma$', color='darkred')
    plt.tight_layout()
    plt.savefig('contour_plot_uncertainties.png',dpi = 500)
    plt.show()

    mass_uncertainty = (max_mass - min_mass)/2
    width_uncertainty = (max_width - min_width)/2
    lifetime_uncertainty = (((pc.hbar/(min_width * pc.electron_volt * 10**9)) -
                             (pc.hbar/(max_width * pc.electron_volt * 10**9)))/2)
    return mass_uncertainty,width_uncertainty,lifetime_uncertainty


def plot_data_fit(filtered_data, optimized_mass, optimized_width, energy, reduced_chi_squared):
    '''
    This function plots the filtered experimental data obtained from
    the detectors. It also plots the minimised chi-square fit of this data.
    Energy is on the x-axis and cross section is on the y-axis.

    Parameters
    ----------
    filtered_data : array
        Filtered experimental data.
    optimized_mass : float
        Optimized mass of the Z boson.
    optimized_width : float
        Optimized width of the Z boson.
    energy : array
        Array of energy values.
    reduced_chi_squared : float
        Reduced chi-squared value
    '''
    if filtered_data is None:
        return None
    energy_range = np.linspace(min(energy), max(energy), 300)
    cross_section_data = filtered_data[:, 1]
    cross_section_curve = find_cross_section(optimized_mass, optimized_width, energy_range)
    error_data = filtered_data[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.errorbar(energy, cross_section_data, yerr=error_data, fmt='o',
                color='#5097a4', label='experimental data', alpha=0.6)
    ax.plot(energy_range, cross_section_curve, label='Fitted Curve', color='darkblue', linewidth=3)
    ax.set_xlabel('Centre-of-Mass Energy(E)-GeV')
    ax.set_ylabel('Cross section(σ)-nb')
    ax.set_title(f'fitted data: $M_z={optimized_mass:.4g}$ ,'\
                 f'$\Gamma_z={optimized_width:.4g}$ , $\chi_r^2={reduced_chi_squared:.3f}$')
    ax.legend()
    ax.grid(True)
    plt.savefig('fitted_Z_boson_data.png',dpi = 500)
    plt.show()


def main():
    '''
    this runs the main code

    Returns
    -------
    None.

    '''
    raw_data = read_and_combine_data()
    if raw_data is None:
        return None
    filtered_data = remove_outliers(raw_data)
    energy = filtered_data[:,0]
    chi_min = minimise_chi_squared(filtered_data,energy)
    reduced_chi_squared = find_reduced_chi_squared(energy,chi_min)
    optimized_mass, optimized_width = chi_min.x

    #here the data is further filtered to remove points that do not fit the curve
    cross_section_data = filtered_data[:, 1]
    cross_section_fit = find_cross_section(optimized_mass, optimized_width, energy)
    error_data = filtered_data[:, 2]
    non_zero_errors = np.where(error_data != 0)[0]#ignoring points with zero error
    condition = np.abs(cross_section_data[non_zero_errors] -
                       cross_section_fit[non_zero_errors])/error_data[non_zero_errors]
    filtered_data = filtered_data[non_zero_errors][condition <= 3]

    #now we rerun the code again with the cleaned data
    energy = filtered_data[:,0]
    chi_min = minimise_chi_squared(filtered_data,energy)
    reduced_chi_squared = find_reduced_chi_squared(energy,chi_min)
    optimized_mass, optimized_width = chi_min.x
    boson_lifetime = find_lifetime(optimized_width)

    #here are our results
    plot_data_fit(filtered_data, optimized_mass, optimized_width,energy,reduced_chi_squared)
    mass_uncertainty, width_uncertainty, lifetime_uncertainty = uncertainty_contour_plot(
        filtered_data, energy, chi_min)
    print(f'the reduced chi-squared value of this fit: χ_r = {reduced_chi_squared:.3f}')
    print(f'Optimized mass of the Z_0 boson: M_z = {optimized_mass:.4g}'\
          f' ± {mass_uncertainty:.3g}(GeV/c^2)   ')
    print(f'Optimized width of the Z_0 boson: Γ_z = {optimized_width:.4g}'\
          f' ± {width_uncertainty:.3g}(GeV)')
    print(f'lifetime of the Z_0 boson: τ_z = {boson_lifetime:.3g}'\
          f' ± {lifetime_uncertainty:.3g}seconds')
if __name__ == "__main__":
    main()
