"""
% combined EQ+tsunami damage assessment

original code written by Mohammad Shafiqual Alam and Andre Barbosa in Matlab.
modified by Sabarethinam Kameshwar.
converted from Matlab to python by Dylan R. Sanderson
Feb. 2019
"""



import os
import sys
import numpy as np
from scipy import io
import scipy.stats as st
import time
import h5py

sys.path.insert(0, '../../py_helpers')
import DC_helper as DCH
from h5_writer import h5_writer as h5wrt
from csv_writer import csv_writer as csvwrt

np.random.seed(1337)



class building_damage_tsu_eq():
	def __init__(self):
		self.setup()
		self.run()


	# ~~~ user input up top ~~~
	def user_input_vals(self):
		self.n_sims = 10000			# number of simulations
		self.retrofit_vals = 1		# originally set to 4

		self.write_h5 = False
		self.write_csv = False
		self.outfiles = ['habitable', 'Data']

	def setup(self):
		self.user_input_vals()
		self.load_h5_files()
		self.initialize_variables()
		self.preallocate_space()


	def run(self):
		self.print_runinfo()
		for self.retrofit in range(self.retrofit_vals):
			print('retrofit: {}' .format(self.retrofit))
			self.define_building_reptime()
			for self.r in range(len(self.RT)):
				timer = time.time()
				self.cases = self.Data[:,4]>0
				for self.i in range(self.n_sims):
					self.building_eval()
				print('elapsed time: {}' .format(time.time() - timer))
			
			# self.reorg_gis_data()
			# self.write_ds_data()
		
	# ~~~ secondary methods ~~~
	#  ~~ from self.setup() ~~
	def load_h5_files(self):
		self.GS_DamageState = self.readh5('building_damage_eq.h5', 'GS_DamageState')
		self.Tsu_DamageState = self.readh5('building_damage_tsu.h5', 'Tsu_DamageState')
		self.Data = DCH.readmat('Data_Seaside_V1.mat', 'Data', dtype = 'array')

	def initialize_variables(self):
		self.RT = np.array([100, 200, 250, 500, 1000, 2500, 5000, 10000])
		self.n_lots = len(self.Data[:,0])	# number of tax lots
		self.n_DS = 5						# number of damage states
		self.outfilename = 'building_damage_tsu_eq'
		self.damage_ratio = np.array([0, 0.005, 0.155, 0.555, 0.9])

	# 	self.BuildingClass_ID = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) 
	# 	self.BuildingClass = np.array(['W1', 'W2', 'C1L', 'C1M', 'C1H', 'C2L', 'C2M', 'C2H', 'C3L', 'C3M', 'C3H', 'S1L', 'S1M', 'S1H'])
	# 	self.BuildingPeriod = np.array([0.35, 0.4, 0.4, 0.75, 1.45, 0.35, 0.56, 1.09, 0.35, 0.56, 1.09, 0.5, 1.08, 2.21])
	# 	self.T = np.array([0.3, 0.4, 0.4, 0.75, 1.5, 0.3, 0.6, 1, 0.3, 0.6, 1, 0.5, 1, 2])  # IM computed at this periods;
	# 	self.g=386.4	# in/s2
	# 	self.Data[np.where(self.Data[:,4] == 2.5)[0], 4] = 3
	# 	self.ori_building_code = self.Data[:,8]
	# 	new_cols = np.zeros(len(self.Data)*23).reshape(len(self.Data), 23)		#preallocating empty space in Data matrix 
	# 	self.Data = np.hstack((self.Data, new_cols))
	# 	self.setup_fragilities_from_hazus()
	# 	self.outfilename = 'building_damage_tsu'

	def preallocate_space(self):
		RT_size = len(self.RT)

		self.time_90p_restore_building = np.zeros(2*self.retrofit_vals*RT_size*self.n_sims).reshape(2, self.retrofit_vals, RT_size, self.n_sims)	# "2" is for "fast_vals"
		self.time_100p_restore_building = np.zeros(2*self.retrofit_vals*RT_size*self.n_sims).reshape(2, self.retrofit_vals, RT_size, self.n_sims)
		self.tot_rep_cost_building_eq = np.zeros(1*self.retrofit_vals*RT_size*self.n_sims).reshape(1, self.retrofit_vals, RT_size, self.n_sims)
		self.tot_rep_cost_building_tsu = np.zeros(1*self.retrofit_vals*RT_size*self.n_sims).reshape(1, self.retrofit_vals, RT_size, self.n_sims)
		self.tot_rep_cost_building = np.zeros(1*self.retrofit_vals*RT_size*self.n_sims).reshape(1, self.retrofit_vals, RT_size, self.n_sims)
		self.frac_damaged_building = np.zeros(1*self.retrofit_vals*RT_size*self.n_sims).reshape(1, self.retrofit_vals, RT_size, self.n_sims)
		self.frac_habitable_building = np.zeros(1*self.retrofit_vals*RT_size*self.n_sims).reshape(1, self.retrofit_vals, RT_size, self.n_sims)
		
		self.rep_cost_buildings_eq = np.zeros(RT_size*len(self.Data[:,0])*self.n_sims).reshape(RT_size, len(self.Data[:,0]), self.n_sims)
		self.rep_cost_buildings_tsu = np.zeros(RT_size*len(self.Data[:,0])*self.n_sims).reshape(RT_size, len(self.Data[:,0]), self.n_sims)
		self.rep_cost_buildings = np.zeros(RT_size*len(self.Data[:,0])*self.n_sims).reshape(RT_size, len(self.Data[:,0]), self.n_sims)

		self.DS_Save_eq = np.zeros((RT_size, len(self.Data[:,0]), self.n_sims))
		self.DS_Save_tsu = np.zeros((RT_size, len(self.Data[:,0]), self.n_sims))
		self.DS_Save = np.zeros((RT_size, len(self.Data[:,0]), self.n_sims))

	#  ~~ from self.run() ~~ 
	def print_runinfo(self):
		print('n_sims: {}' .format(self.n_sims))
		print('retrofit_vals: {}' .format(self.retrofit_vals))
		print('write h5: {}' .format(self.write_h5))
		print('write csv: {}\n' .format(self.write_csv))


	def define_building_reptime(self):
		self.building_rep_time_mu = np.array([0.001*5, 0.5*120, 360, 720] )	# restoration time % table 15.10 and 15.11
		self.building_rep_time_std = np.array([0.5, 0.5, 0.5, 0.5])*self.building_rep_time_mu				# std for repair time
		self.building_rep_time_cov = self.building_rep_time_std/self.building_rep_time_mu	# COV of repair time
		self.building_rep_time_log_med = np.log(self.building_rep_time_mu/np.sqrt(self.building_rep_time_cov**2+1)) # lognormal parameters for repair time model
		self.building_rep_time_beta = np.sqrt(np.log(self.building_rep_time_cov**2+1))
		self.building_rep_time_covm = self.building_rep_time_beta[:, None]*self.building_rep_time_beta

		self.building_rep_time_mu2 = np.array([0.001*5, 0.5*120, 360, 720])*0.5	# restoration time % table 15.10 and 15.11
		self.building_rep_time_std2 = np.array([0.5, 0.5, 0.5, 0.5])*self.building_rep_time_mu2*0.5				# std for repair time
		self.building_rep_time_cov2 = self.building_rep_time_std2/self.building_rep_time_mu2	# COV of repair time
		self.building_rep_time_log_med2 = np.log(self.building_rep_time_mu2/np.sqrt(self.building_rep_time_cov2**2+1)) # lognormal parameters for repair time model
		self.building_rep_time_beta2 = np.sqrt(np.log(self.building_rep_time_cov2**2+1))
		self.building_rep_time_covm2 = self.building_rep_time_beta2[:, None]*self.building_rep_time_beta2


	def building_eval(self):
		DS_tsu = 5-np.sum(np.heaviside(np.random.uniform(low=0, high=1, size=(self.n_lots,1)) - self.Tsu_DamageState[self.retrofit, self.r], 0.5), axis = 1) 	# damage state
		DS_eq = 5-np.sum(np.heaviside(np.random.uniform(low=0, high=1, size=(self.n_lots,1)) - self.GS_DamageState[self.retrofit, self.r], 0.5), axis = 1) 	# damage state
		DS = np.max(np.column_stack([DS_eq, DS_tsu]), axis = 1) 

		self.DS_Save_eq[self.r, :, self.i] = DS_eq
		self.DS_Save_tsu[self.r, :, self.i] = DS_tsu
		self.DS_Save[self.r, :, self.i] = DS

		ind = np.array([(DS_eq==4) & (DS_tsu==4)][0])
		DS[ind] = 5
		ind_count1 = np.sum(ind)
		ind = np.array([(DS_eq==3) & (DS_tsu==3)][0])
		DS[ind] = 4
		ind_count2 = np.sum(ind)

		rep_time_building_all = np.column_stack((np.zeros(self.n_lots), np.exp(np.random.multivariate_normal(self.building_rep_time_log_med,self.building_rep_time_covm,self.n_lots))))	# generating correlated repair time estimates
		rep_time_building_all2 = np.column_stack((np.zeros(self.n_lots), np.exp(np.random.multivariate_normal(self.building_rep_time_log_med2,self.building_rep_time_covm2,self.n_lots))))	# generating correlated repair time estimates

		rep_time_buildings = np.array([rep_time_building_all[int(i), int(obj)-1] for i, obj in enumerate(DS)])
		rep_time_buildings2 = np.array([rep_time_building_all2[int(i), int(obj)-1] for i, obj in enumerate(DS)])

		self.time_90p_restore_building[0, self.retrofit, self.r, self.i] = np.percentile(rep_time_buildings[self.cases], 90)
		self.time_100p_restore_building[0, self.retrofit, self.r, self.i] = np.max(rep_time_building_all[self.cases])
		
		self.time_90p_restore_building[1, self.retrofit, self.r, self.i] = np.percentile(rep_time_buildings2[self.cases], 90)
		self.time_100p_restore_building[1, self.retrofit, self.r, self.i] = np.max(rep_time_building_all2[self.cases])

		rep_costs_buildings_eq = self.damage_ratio[DS_eq.astype(int)-1] * self.Data[:,10]
		rep_costs_buildings_tsu = self.damage_ratio[DS_tsu.astype(int)-1] * self.Data[:,10]
		rep_costs_buildings = self.damage_ratio[DS.astype(int)-1] * self.Data[:,10]

		self.tot_rep_cost_building_eq[0, self.retrofit, self.r, self.i] = np.sum(rep_costs_buildings_eq[self.cases])
		self.tot_rep_cost_building_tsu[0, self.retrofit, self.r, self.i] = np.sum(rep_costs_buildings_tsu[self.cases])
		self.tot_rep_cost_building[0, self.retrofit, self.r, self.i] = np.sum(rep_costs_buildings[self.cases])
		
		self.habitable = np.zeros(np.size(DS))
		self.habitable[DS<=3] = 1
		self.habitable[DS>=4] = 0

		self.frac_habitable_building[0, self.retrofit, self.r, self.i] = np.mean(self.habitable[self.cases])
		self.frac_damaged_building[0, self.retrofit, self.r, self.i] = np.mean(DS[self.cases]>2)

		# note: drs, the below is new
		self.rep_cost_buildings_eq[self.r, :, self.i] = rep_costs_buildings_eq[:]
		self.rep_cost_buildings_tsu[self.r, :, self.i] = rep_costs_buildings_tsu[:]
		self.rep_cost_buildings[self.r, :, self.i] = rep_costs_buildings[:]
		
	def reorg_gis_data(self):
		gis_data = np.zeros((len(self.Data[:,0]), len(self.RT)*3 + 1))
		gis_data[:,0] = self.Data[:,2]
		header = np.empty(len(self.RT)*3 + 1, dtype = object)
		for r in range(len(self.RT)):
			data_temp_eq = self.rep_cost_buildings_eq[r]
			data_temp_tsu = self.rep_cost_buildings_tsu[r]
			data_temp = self.rep_cost_buildings[r]
			
			avg_eq = np.average(data_temp_eq, axis = 1)
			avg_tsu = np.average(data_temp_tsu, axis = 1)
			avg = np.average(data_temp, axis = 1)

			gis_data[:,r+1] = avg_eq
			gis_data[:,r+9] = avg_tsu
			gis_data[:,r+17] = avg

			header[r+1] = "eq_" + str(self.RT[r])
			header[r+9] = 'tsu_' + str(self.RT[r])
			header[r+17] = 'tsu_eq_' + str(self.RT[r])

		header[0] = 'ID'
		header = list(header)
		header = ', '.join(header)

		csv_str = 'building_damage_med_{}' .format(self.retrofit)
		csvwrt(gis_data, header, csv_str)


	def write_ds_data(self):
		ds_eq_save = np.zeros((len(self.Data[:,0]), len(self.RT)*self.n_DS+1))
		ds_tsu_save = np.zeros((len(self.Data[:,0]), len(self.RT)*self.n_DS+1))
		ds_save = np.zeros((len(self.Data[:,0]), len(self.RT)*self.n_DS+1))

		ds_eq_save[:, 0] = self.Data[:,2]
		ds_tsu_save[:, 0] = self.Data[:,2]
		ds_save[:, 0] = self.Data[:,2]

		count = 1
		header = np.empty(len(self.RT)*self.n_DS+1, dtype = object)

		for r in range(len(self.RT)):
			for i in range(self.n_DS):
				key = 'RT_{}_DS_{}' .format(self.RT[r], i)
				ds_eq_save[:, count] = np.sum((self.DS_Save_eq[r, :, :] == i+1), axis = 1)/self.n_sims
				ds_tsu_save[:, count] = np.sum((self.DS_Save_tsu[r, :, :] == i+1), axis = 1)/self.n_sims
				ds_save[:, count] = np.sum((self.DS_Save[r, :, :] == i+1), axis = 1)/self.n_sims
				header[count] = key
				count += 1

		header[0] = 'ID'
		header = list(header)
		header = ', '.join(header)

		csvwrt(ds_eq_save, header, 'Building_DS_Eq')
		csvwrt(ds_tsu_save, header, 'Building_DS_Tsu')
		csvwrt(ds_save, header, 'Building_DS')
		

	# ~~~ tertiary and beyond methods ~~~
	def readh5(self, file, key):
		f = h5py.File(file, 'r')
		return np.array(f[key])




bdte = building_damage_tsu_eq()
if bdte.__dict__['write_h5'] == True:
	h5wrt(bdte)
if bdte.__dict__['write_csv'] == True:
	csvwrt(bdte)