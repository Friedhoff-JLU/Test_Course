'''This file contains the functions used in the notebook "Fitting a function to noisy data"'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.optimize import curve_fit
import ipywidgets as widgets
from IPython.display import display
import random


def plot_data_with_noise(a, b,noise_level, n, func):
    '''This function is used to plot the data with noise and the noise distribution
    a: float from main notebook
    b: float from main notebook
    noise_level: float from a slider
    n: int from a slider
    func: function
    '''
    x = np.linspace(0, 10,n)
    y = func(x,a,b)
    # noise = np.random.normal(loc=0.0, scale= y.max()/100* noise_level, size=len(x))
    # alternatively, you can use the following line to generate the noise using the random module
    noise = [random.gauss(0, y.max()/100* noise_level) for i in range(len(x))]
    y_exp = y + noise
    fig, axs = plt.subplots(2,1, figsize=(4, 4), gridspec_kw={'height_ratios': [3, 1]})
    # set the size of subplots to 6,8 and 6,4
    axs[0].plot(x, y, label='data without noise', color = 'black')
    axs[0].scatter(x, y_exp, label='data with noise')
    # plot mean and standard deviation in the noise subplot
    axs[1].plot(x, np.mean(noise)*np.ones(len(x)), label='mean', color = 'black')
    axs[1].plot(x, np.mean(noise)+2*np.std(noise)*np.ones(len(x)), label='mean + std', color = 'red')
    axs[1].plot(x, np.mean(noise)-2*np.std(noise)*np.ones(len(x)), label='mean - std', color = 'red')
    axs[1].scatter(x, noise, label= 'noise')
    axs[0].set_ylabel('y')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('noise')
    
    axs[0].legend()
    plt.tight_layout()
    plt.show()
    # make a histogram of the noise
    fig, axs = plt.subplots(1,1, figsize=(4, 4))
    axs.hist(noise, bins = 10)


def show_plot(a, b, func):
    '''This function is used to display the plot widget in the notebook
    a: float
    b: float
    func: function
    '''
    func = func
    noise_level_slider = widgets.FloatSlider(value=5, min =0, max = 50, step =1 , description='noise level', readout_format='.0f')
    n_slider = widgets.IntSlider(value=20, min =3, max = 500, step =1 , description='n', readout_format='.0f')
    plot_widget = widgets.interactive(plot_data_with_noise,  a=widgets.fixed(a), b=widgets.fixed(b), noise_level = noise_level_slider, n = n_slider, func=widgets.fixed(func))
    display(plot_widget)
    return n_slider,  noise_level_slider

def perform_fit2(a_init, b_init, func, x, y_exp):
    # plot data
    # fit the function
    popt, pcov = curve_fit(func, x, y_exp, p0=(a_init, b_init))
    # print the values for a and b 
    print(f'a_init = {a_init:.1f}')
    print(f'b_init = {b_init:.1f}')
    # print(f'a = {a:.1f}')
    # print(f'b = {b:.1f}')
    # print(f'noise level = {noise_level:.1f} %')
    # print the fit parameters
    print(f'a_fit =, {popt[0]:.2f} +/- {np.sqrt(pcov[0,0]):.2f}')
    print(f'b_fit = {popt[1]:.2f} +/- {np.sqrt(pcov[1,1]):.2f}')

    # plot the fit
    # plt.scatter(x, y, label='data')
    fig, axs = plt.subplots(3,1, figsize=(6, 6), gridspec_kw={'height_ratios': [2,1, 2]})
    # color the point in raibow colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(x)))
    axs[0].scatter(x, y_exp, c=colors, label='data with noise')
    # include the the error of the fit parameters in the legend
    axs[0].plot(x, func(x, a_init, b_init), 'b--', label='initial guess')
    axs[0].plot(x, func(x, popt[0]+pcov[0,0], popt[1]+pcov[1,1]), 'r--', label='fit: a=%5.2f, b=%5.2f' % tuple(popt+pcov[0,0]))
    colors2 = colors[1:]
    # plot the residuals
    axs[1].scatter(x, y_exp-func(x, popt[0], popt[1]), c=colors, label='residuals')
    axs[2].scatter(1/x[1:], 1/y_exp[1:],  c=colors2, label='data with noise')
    axs[2].plot(1/x[1:], 1/func(x[1:], popt[0], popt[1]), 'r--', label='fit: a=%5.2f, b=%5.2f' % tuple(popt))
    axs[2].plot(1/x[1:], 1/func(x[1:], a_init, b_init), 'b--', label='initial guesss')
    
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title("Direct plot")
    axs[2].set_xlabel('1/x')
    axs[2].set_ylabel('1/y')
    axs[2].set_title("Lineweaver-Burk plot")
    # show the plot
    axs[0].legend()
    # layout tight
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    display(plot_widget)