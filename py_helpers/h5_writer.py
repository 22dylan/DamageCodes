"""
an h5 writer file. 
objects from transportation, epn, water, and building codes are passed in (e.g. "item")
	-dictionary is isolated from this file
	-all variables in dictionary are written to an h5 file (except for graph object)

dylan sanderson
feb. 7th, 2018
oregon state university
"""

import numpy as np
import h5py
import time
import os
import igraph


class h5_writer():
	def __init__(self, item):
		self.data_dict = item.__dict__
		self.create_h5()
		self.write_h5()
		self.close_h5file()

	def create_h5(self):
		path = os.path.join(os.getcwd(), 'output')			# create relative path name
		if not os.path.isdir(path):							# create path if it doesn't exist
			os.mkdir(path)

		timestr = time.strftime("%Y%m%d-%H%M%S")			# setting up a time string
		filename = '{}_{}.h5' .format(self.data_dict['outfilename'], timestr)	# creating a filename with the time string
		filename = os.path.join(path, filename)				# joining output directory to filename
		self.h5file = h5py.File(filename, 'w')				# creates h5 file
		self.single_group = self.h5file.create_group('single_values')	#creates a group for the single values (e.g. strings, floats, and ints)

	def write_h5(self):
		for self.key in self.data_dict.keys():
			temp_data = self.data_dict[self.key]

			if isinstance(temp_data, (int, str, float)):	# looking for single values (e.g. int, string, float). for organization purposes withing h5 file
				self.write_single_val()
			elif isinstance(temp_data, (np.ndarray)):	# looking for arrays or lists
				self.write_array()
			elif isinstance(temp_data, list):
				self.write_list()
			elif isinstance(temp_data, dict):
				self.write_dict()
			elif isinstance(temp_data, (igraph.Graph)):		# looking for graph item (skips this)
				pass
			else:
				print('No output writer for variables of type: {}' .format(type(temp_data)))

	def write_single_val(self):
		dset_out = self.single_group.create_dataset(self.key, data = self.data_dict[self.key]) 	# writing data to h5 (within single values group)

	def write_array(self):	
		if self.data_dict[self.key].dtype == '<U3':		# checking if datatype is unicode (h5py can't handle these)
			self.data_dict[self.key] = self.data_dict[self.key].astype('<S3')	# converting to ascii if so
		dset_out = self.h5file.create_dataset(self.key, data = self.data_dict[self.key])		# writing data to h5

	def write_list(self):
		print('note: drs, need to figure out how to write list to h5.')
		# dset_out = self.h5file.create_dataset(self.key, data = self.data_dict[self.key])

	def write_dict(self):
		print('note: drs, need to look into writing dict to h5file. store a dict as a single h5 group, loop through keys, and etc.')

	def close_h5file(self):
		self.h5file.close()		# self-explanatory



