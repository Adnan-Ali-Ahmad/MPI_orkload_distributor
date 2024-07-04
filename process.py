# flake8: noqa

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import os
import re
import sys
import utils as utils
import osyris


class Process:
	"""
	Data structure that handles MPI threads and contains RAMSES data.
	"""
	def __init__(self, mpi_comm, f_read=1, skip=0, skip_end=-1, ramses_ism=True):
		self.comm = mpi_comm
		self.world_size = self.comm.size
		self.rank = self.comm.Get_rank()
		self.f_read = f_read
		self.skip = skip
		self.skip_end = skip_end
		self.ramses_ism = ramses_ism


	def book_making(self):
		"""
		Asks process 0 to create output dirs if they do not exist.
		"""
		sim_name = self.ramses_data_dir.split("/")[-2]
		self.osyris_results_dir = self.output_folder + sim_name + "/"
		self.osyris_log_dir = self.log_outputs_dir + "runtime_log_" + sim_name + "/"
		self.osyris_dump_dir = self.dump_dir + sim_name + "/"
		if self.rank == 0:
			print("{:=^100}".format("  Process 0 proceeding to book making  "))
			isExist_outputs_dir = os.path.exists(self.osyris_results_dir)
			isExist_log_dir = os.path.exists(self.osyris_log_dir)
			isExist_data_dump_dir = os.path.exists(self.osyris_dump_dir)
			if not isExist_outputs_dir:
				os.makedirs(self.osyris_results_dir)
				print("Created output folder for simulation " + sim_name)
			else:
				print("Output folder for simulation  " + sim_name + " already exists.")
			if not isExist_log_dir:
				os.makedirs(self.osyris_log_dir)
				print("Created log folder for simulation " + sim_name)
			else:
				print("log folder for simulation " + sim_name + " already exists.")
			if not isExist_data_dump_dir:
				os.makedirs(self.osyris_dump_dir)
				print("Created data dump folder for simulation " + sim_name)
			else:
				print("Data dump folder for simulation " + sim_name + " already exists.")
			print("{:=^100}".format(" Book making completed "))


	def set_path(self, path=None):
		"""
		Set the path to the RAMSES outputs.
		"""
		try:
			if path.endswith("/"):
				self.ramses_data_dir = path
			else:
				self.ramses_data_dir = path + "/"
		except AttributeError:
			print("Process instance needs a path to RAMSES data!")


	def set_log_dir(self, path=None):
		"""
		Set the path of log results produced by the process. (eg. max density, temp info ...)
		"""
		try:
			if path.endswith("/"):
				self.log_outputs_dir = path
			else:
				self.log_outputs_dir = path + "/"
		except AttributeError:
			print("Process instance needs a valid output path to produce results!")


	def set_data_dump_dir(self, path=None):
		"""
		Set the path of data dumps produced by the thread
		"""
		try:
			if path.endswith("/"):
				self.dump_dir = path
			else:
				self.dump_dir = path + "/"
		except AttributeError:
			print("Process instance needs a valid output path to produce results!")


	def set_results_dir(self, path=None):
		"""
		Set the path of results produced by the process.
		"""
		try:
			if path.endswith("/"):
				self.output_folder = path
			else:
				self.output_folder = path + "/"
		except AttributeError:
			print("Process instance needs a valid output path to produce results!")


	def probe_outputs(self):
		"""
		Fetch all RAMSES output folders.
		"""
		try:
			self.outputs = [file for file in os.listdir(self.ramses_data_dir) if "output" in file and "ramses" not in file] #Probing all RAMSES output folders
		except AttributeError:
			print("Path to simulation has not been assigned, cannot proceed.")
			exit(1)


	def assign_workload(self):
		"""
		Probe outputs and select folders to read.
		"""
		self.probe_outputs()
		self.outputs_id = [int(re.findall(r'\d+',file)[0]) for file in self.outputs] #integer list of all output folders
		self.outputs_id = [o for i,o in enumerate(self.outputs_id) if i%self.f_read == 0] #grab every f_read output folder
		self.outputs_id.sort() #sorting

		i0 = self.outputs_id[0] if self.skip!=0 else 0
		if self.skip_end == -1:
			self.skip_end = self.outputs_id[-1]

		self.outputs_id = self.outputs_id[self.skip-i0:self.skip_end] # skip first "skip" folders
		self.sim_nout = len(self.outputs_id) # number of output folders in simulation
		self.proc_nout = int(self.sim_nout/self.world_size) # number of output folders assigned to process

		if self.rank == 0: # ask first process to read all remaining folders since it finishes early
			tmp = self.outputs_id #tmp variable
			self.outputs_id = tmp[self.proc_nout*self.rank:self.proc_nout*(self.rank+1)] # new list of outputs the thread is going to process
			self.outputs_id += tmp[self.proc_nout*(self.world_size):] #grab remaining folders
		else:
			self.outputs_id = self.outputs_id[self.proc_nout*self.rank:self.proc_nout*(self.rank+1)] # new list of outputs the thread is going to process

		self.output_folder = '' # file being currently read by process


	def load_ramses_data(self, output_id, path=None):
		if path is None:
			path = self.ramses_data_dir
		self.ramses_data = osyris.Dataset(output_id, path=path).load(ramses_ism=self.ramses_ism)


	def execute_instructions(self, instruction_list):
		"""
		Loop over workload and execute user-assigned tasks.
		"""
		for j,i in enumerate(self.outputs_id):
			self.j = j
			self.i = i
			self.output_folder = self.ramses_data_dir+('output_%05d' % (i))
			s = "Process {:2d}/{:2d} currently reading {: <80} Folders remaining: {}".format(self.rank, self.world_size-1, self.output_folder, len(self.outputs_id)-j)
			print(s)
			with utils.suppress_stdout():
				self.load_ramses_data(i)
				for func in instruction_list:
					func(self)
			del self.ramses_data # freeing memory
		print("{:=^{}}".format(" Process {} completed all tasks ".format(self.rank), len(s)))
