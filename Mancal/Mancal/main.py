import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import tkinter as tk
from tkinter import *

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import data
from Mancal import run_Complete_Training_Model, run_Show_Results

def run_Model(dir,ndatapoints,ntrain):

    run_Complete_Training_Model(dir,ndatapoints,ntrain)

def run_Results(dir):

    run_Show_Results(dir)


### THIS PART OF THE CODE JUST BUILD THE GRAPHICAL USER INTERFACE

def main():

    window = tk.Tk()

    greeting = tk.Label(text="Time Series Classification (TSC)")
    greeting.grid(row=1,column=1)

    dir_label = tk.Label(window,text="Directory Name")
    dir_label.grid(row=2,column=1)
    dir_entry = tk.Entry(window)
    dir_entry.grid(row=2,column=2)

    dataPoints_label = tk.Label(window,text="# of Data Points")
    dataPoints_label.grid(row=3,column=1)
    dataPoints_entry = tk.Entry(window)
    dataPoints_entry.grid(row=3,column=2)

    ntrain_label = tk.Label(window,text="# of Training Time Series (max=11)")
    ntrain_label.grid(row=4,column=1)
    ntrain_entry = tk.Entry(window)
    ntrain_entry.grid(row=4,column=2)

    button = tk.Button(
        window,
        text="Run Model",
        width=12,
        height=3,
        bg="white",
        fg="black",
        command = (lambda: run_Model(dir_entry.get(),dataPoints_entry.get(),ntrain_entry.get()))
    ).grid(row=5,column=1)

    button = tk.Button(
        window,
        text="Show Results",
        width=12,
        height=3,
        bg="white",
        fg="black",
        command = (lambda: run_Results(dir_entry.get()))
    ).grid(row=5,column=2)

    
    window.mainloop()

main()