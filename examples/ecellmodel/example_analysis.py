import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import glob
import os
from ovito.io import import_file  
import ovito.modifiers as om
import lammpskit.ecellmodel.filament_layer_analysis as lmpfl


output_dir =  os.path.join("..", "..","output", "ecellmodel")

## The following code block generates plots that track the evolution of the 
# filament connectivity state, gap, and separation over time for each 
# timeseries trajectory file in the file_list.

## Simulation parameters corresponding to the respective raw data
TIME_STEP = 0.001
DUMP_INTERVAL_STEPS = 500

MIN_SIM_STEP = 0
MAX_SIM_STEP = 500000
loop_start = int(MIN_SIM_STEP / DUMP_INTERVAL_STEPS)
loop_end = int(MAX_SIM_STEP / DUMP_INTERVAL_STEPS)

time_points = np.linspace(loop_start*DUMP_INTERVAL_STEPS*TIME_STEP,loop_end*DUMP_INTERVAL_STEPS*TIME_STEP,loop_end-loop_start+1)
print(np.shape(time_points),'\n',time_points[-1])
###################################

data_path = os.path.join("..","..", "data","ecellmodel", "processed","trajectory_series", "*.lammpstrj")
analysis_name = 'track_'
# data_path = "*.lammpstrj"
unsorted_file_list = glob.glob(data_path)
file_list = sorted(unsorted_file_list)
print('The data_path is ',data_path)
print(analysis_name,file_list)

lmpfl.track_filament_evolution(file_list, analysis_name,TIME_STEP,DUMP_INTERVAL_STEPS,output_dir=output_dir)

## The following code block generates plots of atomic distributions
# and compares the displacements of Hf, O, and Ta for different temperatures

## Simulation parameters corresponding to the respective raw data
TIME_STEP = 0.0002
DUMP_INTERVAL_STEPS = 5000

MIN_SIM_STEP = 0
MAX_SIM_STEP = 2500000
loop_start = int(MIN_SIM_STEP / DUMP_INTERVAL_STEPS)
loop_end = int(MAX_SIM_STEP / DUMP_INTERVAL_STEPS)

time_points = np.linspace(loop_start*DUMP_INTERVAL_STEPS*TIME_STEP,loop_end*DUMP_INTERVAL_STEPS*TIME_STEP,loop_end-loop_start+1)
print(np.shape(time_points),'\n',time_points[-1])
SKIP_ROWS_COORD= 9   
HISTOGRAM_BINS = 15
###################################

analysis_name = f'temp_{HISTOGRAM_BINS}'
data_path =  os.path.join("..", "..","data","ecellmodel", "raw", "temp*.lammpstrj")
unsorted_file_list = glob.glob(data_path)
file_list = sorted(unsorted_file_list)
print(analysis_name,file_list)
labels = ['300 K','900 K', '1300 K']
lmpfl.plot_atomic_distribution(file_list,labels,SKIP_ROWS_COORD,HISTOGRAM_BINS,analysis_name,output_dir=output_dir)

data_path =  os.path.join("..", "..","data","ecellmodel", "raw", "*K_Hfmobilestc1.dat")
analysis_name = f'displacements_temp_Hf'
unsorted_file_list = glob.glob(data_path)
file_list = sorted(unsorted_file_list)
print(analysis_name,file_list)
labels = ['300 K','900 K', '1300 K']
lmpfl.plot_displacement_comparison(file_list, loop_start, loop_end, labels, analysis_name, repeat_count=0,output_dir=output_dir)

data_path =  os.path.join("..", "..","data","ecellmodel", "raw", "*K_Omobilestc1.dat")
analysis_name = f'displacements_temp_O'
unsorted_file_list = glob.glob(data_path)
file_list = sorted(unsorted_file_list)
print(analysis_name,file_list)
labels = ['300 K','900 K', '1300 K']
lmpfl.plot_displacement_comparison(file_list, loop_start, loop_end, labels, analysis_name, repeat_count=0,output_dir=output_dir)

data_path =  os.path.join("..", "..","data","ecellmodel", "raw", "*K_Tamobilestc1.dat")
analysis_name = f'displacements_temp_Ta'
unsorted_file_list = glob.glob(data_path)
file_list = sorted(unsorted_file_list)
print(analysis_name,file_list)
labels = ['300 K','900 K', '1300 K']
lmpfl.plot_displacement_comparison(file_list, loop_start, loop_end, labels, analysis_name, repeat_count=0,output_dir=output_dir)


## The following code block generates plots of atomic and charge distributions 
# and compares the displacements of Hf, O, and Ta for different temperatures   
    ## Simulation parameters corresponding to the respective raw data
TIME_STEP = 0.0002
DUMP_INTERVAL_STEPS = 5000

MIN_SIM_STEP = 0
MAX_SIM_STEP = 2500000
loop_start = int(MIN_SIM_STEP / DUMP_INTERVAL_STEPS)
loop_end = int(MAX_SIM_STEP / DUMP_INTERVAL_STEPS)

time_points = np.linspace(loop_start*DUMP_INTERVAL_STEPS*TIME_STEP,loop_end*DUMP_INTERVAL_STEPS*TIME_STEP,loop_end-loop_start+1)
print(np.shape(time_points),'\n',time_points[-1])
SKIP_ROWS_COORD= 9   
HISTOGRAM_BINS = 15
###################################

analysis_name = f'local_{HISTOGRAM_BINS}'
data_path =  os.path.join("..", "..","data","ecellmodel", "raw", "local2*.lammpstrj")
unsorted_file_list = glob.glob(data_path)
file_list = sorted(unsorted_file_list)
print(analysis_name,file_list)
labels = ['initial','final']
lmpfl.plot_atomic_distribution(file_list,labels,SKIP_ROWS_COORD,HISTOGRAM_BINS,analysis_name,output_dir=output_dir)

data_path = os.path.join("..", "..","data","ecellmodel", "raw", "[1-9][A-Z][A-Za-z]mobilestc1.dat")
analysis_name = f'displacements_atom_type'
unsorted_file_list = glob.glob(data_path)
file_list = sorted(unsorted_file_list)
print(analysis_name,file_list)
labels = ['Hf','O', 'Ta']
lmpfl.plot_displacement_comparison(file_list, loop_start, loop_end, labels, analysis_name, repeat_count=0,output_dir=output_dir)


analysis_name = f'local_charge_{HISTOGRAM_BINS}'
data_path =  os.path.join("..", "..","data","ecellmodel", "raw", "local2*.lammpstrj")
unsorted_file_list = glob.glob(data_path)
file_list = sorted(unsorted_file_list)
print(analysis_name,file_list)
labels = ['initial','final']
lmpfl.plot_atomic_charge_distribution(file_list,labels,SKIP_ROWS_COORD,HISTOGRAM_BINS,analysis_name,output_dir=output_dir)