import numpy as np
import matplotlib
matplotlib.use('Agg')
import glob
import os

import lammpskit.ecellmodel.filament_layer_analysis as lmpfl


output_dir =  os.path.join("..", "..","output", "ecellmodel")

## The following code block generates plots of atomic and charge distributions 
# and compares the displacements of Hf, O, and Ta for different temperatures  
#  
## Simulation parameters corresponding to the respective raw data
###################################

TIME_STEP = 0.001
DUMP_INTERVAL_STEPS = 500

MIN_SIM_STEP = 0
MAX_SIM_STEP = 500000

loop_start = int(MIN_SIM_STEP / DUMP_INTERVAL_STEPS)
loop_end = int(MAX_SIM_STEP / DUMP_INTERVAL_STEPS)

time_points = np.linspace(loop_start*DUMP_INTERVAL_STEPS*TIME_STEP,loop_end*DUMP_INTERVAL_STEPS*TIME_STEP,loop_end-loop_start+1)
print(np.shape(time_points),'\n',time_points[-1])
SKIP_ROWS_COORD= 9   
HISTOGRAM_BINS = 15
###################################

analysis_name = f'forming_{HISTOGRAM_BINS}'
data_path =  os.path.join("..", "..","data","ecellmodel", "raw", "[1-9]forming*.lammpstrj")
unsorted_file_list = glob.glob(data_path)
file_list = sorted(unsorted_file_list)
print(analysis_name,file_list)
labels = ['relaxed0V','formed2V']
lmpfl.plot_atomic_distribution(file_list,labels,SKIP_ROWS_COORD,HISTOGRAM_BINS,analysis_name,output_dir=output_dir)

analysis_name = f'post_forming_{HISTOGRAM_BINS}'
data_path =  os.path.join("..", "..","data","ecellmodel", "raw", "[1-9]formed*.lammpstrj")
unsorted_file_list = glob.glob(data_path)
file_list = sorted(unsorted_file_list)
print(analysis_name,file_list)
labels = ['formed2V','formed0V']
lmpfl.plot_atomic_distribution(file_list,labels,SKIP_ROWS_COORD,HISTOGRAM_BINS,analysis_name,output_dir=output_dir)

analysis_name = f'set_{HISTOGRAM_BINS}'
data_path =  os.path.join("..", "..","data","ecellmodel", "raw", "[1-9]set*.lammpstrj")
unsorted_file_list = glob.glob(data_path)
file_list = sorted(unsorted_file_list)
print(analysis_name,file_list)
labels = ['formed0V','set-0.1V']
lmpfl.plot_atomic_distribution(file_list,labels,SKIP_ROWS_COORD,HISTOGRAM_BINS,analysis_name,output_dir=output_dir)

analysis_name = f'break_{HISTOGRAM_BINS}'
data_path =  os.path.join("..", "..","data","ecellmodel", "raw", "[1-9]break*.lammpstrj")
unsorted_file_list = glob.glob(data_path)
file_list = sorted(unsorted_file_list)
print(analysis_name,file_list)
labels = ['set-0.1V','break-0.5V']
lmpfl.plot_atomic_distribution(file_list,labels,SKIP_ROWS_COORD,HISTOGRAM_BINS,analysis_name,output_dir=output_dir)
