"""
a csv writer. 
objects from transportation, epn, water, and building codes are passed in as a 2d matrix
	-matrix: 2d matrix of data
	-header_vals: header string to go with matrix

dylan r. sanderson
feb. 15th, 2018
oregon state university
"""

import os
import numpy as np
import time

class csv_writer():
	def __init__(self, matrix, header_vals, filename):
		self.create_outpath()
		self.write_csv(matrix, header_vals, filename)

	def create_outpath(self):
		self.pathname = os.path.join(os.getcwd(), 'output')				# create relative path name
		if not os.path.isdir(self.pathname):							# create path if it doesn't exist
			os.mkdir(self.pathname)

	def write_csv(self, matrix, header_vals, filename):
		filename = '{}.csv' .format(filename)
		filename = os.path.join(self.pathname, filename)
		frmt = ['%i, ' + '%f, '*(np.shape(matrix)[1]-2)+ '%f']
		np.savetxt(filename, matrix, header = header_vals, delimiter = ',', fmt=frmt[0])

