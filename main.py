# flake8: noqa

import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sbp
from mpi4py import MPI
import os
import re
import process as pc
from instructions import * # import all the funcs from the instructions.py file
import osyris

path = "" # path to simulation data
output_folder = "" # where to save this pipeline's results
log_dir = "" # where to store the log files of this pipeline
data_dump_dir = "" # where to dump this pipeline's heavy data

comm = MPI.COMM_WORLD  # init mpi

thread = pc.Process(comm, skip=0, skip_end=None, f_read=1)
thread.set_path(path=path)
thread.set_results_dir(output_folder)
thread.set_log_dir(log_dir)
thread.set_data_dump_dir(data_dump_dir)
thread.book_making()
thread.assign_workload()
thread.execute_instructions([task1])
