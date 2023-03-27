'''This file contains the functions used in the notebook "Fitting a function to noisy data"'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.optimize import curve_fit
import ipywidgets as widgets
from IPython.display import display
import random

def plot_data_with_noise(noise_level, n):
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


noise_level_slider = widgets.FloatSlider(value=5, min =0, max = 50, step =1 , description='noise level', readout_format='.0f')
n_slider = widgets.IntSlider(value=20, min =3, max = 500, step =1 , description='n', readout_format='.0f')

plot_widget = widgets.interactive(plot_data_with_noise,  noise_level = noise_level_slider, n = n_slider)


if __name__ == '__main__':
    display(plot_widget)