"""
% this code performs analysis of the water network in Seaside subjected to
% earthquakes
% The main objective is to determine the connectivty of the tax lot
% locations to the electric power substation

original code written by Sabarethinam Kameshwar in Matlab
converted from Matlab to python by Dylan R. Sanderson
Jan. 2019
"""

import os
import sys
sys.path.insert(0, '../../py_helpers')
import DC_helper as DCH
from h5_writer import h5_writer as h5wrt
from csv_writer import csv_writer as csvwrt

import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import time
np.random.seed(1337)

class EPN_damage_tsu():
	def __init__(self):
		self.setup()
		self.run()

	# ~~~ user input up top ~~~
	def user_input_vals(self):
		self.n_sims = 10000			# number of simulations
		self.fast_vals = 1								# originally set to 2
		self.retrofit_vals = 1							# originally set to 2

		self.plot_tf = False
		self.write_tf = False

	# ~~~ main methods ~~~
	def setup(self):
		self.user_input_vals()
		self.load_mat_files()
		self.initialize_variables()
		self.preallocate_space()
		self.G = DCH.create_graph(self.adj_EPN)

	def run(self):
		self.print_runinfo()
		for self.fast in range(self.fast_vals):
			print('fast: {}' .format(self.fast))

			for self.retrofit in range(self.retrofit_vals):
				print('retrofit: {}' .format(self.retrofit))
				self.define_pole_rep_variables()
				self.define_pole_frag_variables()
				
				for self.r in range(len(self.RT)): 		#loop through return intervals
					self.define_IMs()
					self.define_flowspeeds()
					self.redefine_pole_frag_variables()

					timer = time.time()
					for self.i in range(self.n_sims):
						self.conn_analysis1()
						self.time_eval()
						self.substation_conn(init_loop = True)
						self.unique_reptime_poles = np.unique(self.rep_time_poles[self.rep_time_poles>0]).astype(int)

						for self.t in range(len(self.unique_reptime_poles)):
							self.conn_analysis2()
							self.time_eval()
							self.substation_conn(init_loop = False)

							if  np.sum(self.down_poles_l) == 0:
								self.complete_loop()
								break
					
						self.time_to_p_conn_eval()
					print('elapsed time: {}' .format(time.time() - timer))

				if self.plot_tf == True:
					self.plot_results()

		# ~~~ damage to water treatment plants ~~~
		self.define_SS_IMs()
		for self.fast in range(self.fast_vals):
			for self.retrofit in range(self.retrofit_vals):
				# self.define_SS_frag_variables()
				self.define_SS_rep_variables()
				
				for self.r in range(len(self.RT)): 		#loop through return intervals
					if self.retrofit == 1:				#if the SS is elevatred by 5 ft
						self.IM_SS_d[self.r] = self.IM_SS_d[self.r] - 5/3.2808
					self.define_flowspeeds_SS()
					self.define_SS_frag_variables()
					self.wtp_eval()
					

	# ~~~ secondary methods ~~~
	#  ~~ from self.setup() ~~
	def load_mat_files(self):
		self.node_IM_tsu = self.readmat('node_IM_tsu.mat', 'node_IM_tsu', dtype = 'array')
		
		self.start_node = self.readmat('EPN_adjacency.mat', 'start_node', dtype = 'array_flat', idx_0 = True)
		self.end_node = self.readmat('EPN_adjacency.mat', 'end_node', dtype = 'array_flat', idx_0 = True)
		self.adj_EPN = self.readmat('EPN_adjacency.mat', 'adj_EPN', dtype = 'adj_list')

		self.tax_lot_info_raw = self.readmat('tax_lot_info.mat', 'tax_lot_info', dtype = 'array')

		self.node_ID = self.readmat('node_pole_info.mat', 'node_ID', dtype = 'array_flat', idx_0 = True)
		self.pole_ID = self.readmat('node_pole_info.mat', 'pole_ID', dtype = 'array_flat', idx_0 = True)
		self.pole_y_n = self.readmat('node_pole_info.mat', 'pole_y_n', dtype = 'array_flat')

		self.misc_data_conversions()

	def initialize_variables(self):
		self.n_nodes = np.shape(self.adj_EPN)[0]
		self.n_links = len(self.start_node)
		self.link_ID = np.array(range(0,self.n_links))

		self.RT = np.array([100, 200, 250, 500 ,1000, 2500, 5000, 10000]) # return period of earthquake events (in years)
		self.time_inst = np.array(range(3*365))+1		# time instances (or, in this case, # days in 3 years)

		# node numbers of water pumps in the water network
		self.wps_nodes = np.array([0, 317, 243]) 	# pumps 1,2 and 3 respectively
		self.wtp_node = np.array([65])
		self.wwtp_node = np.array([108])
		self.lift_st_nodes = np.array([7, 34, 61, 89, 118, 125, 215, 237, 275])
		self.SS_node = np.array([210])

		# cost estimates from HAZUS - in 1000's of USD
		self.pole_cost = np.array([3])
		self.substation_cost = np.array([10000])

		self.damage_ratio_poles_tsu = np.array([0.0, 0.02, 0.10, 0.5, 1.0]) 
		self.frag_mod_fac_SS = np.array([[1.0, 1.5, 2.0], [1.5, 2.5, 3.0]])	# factors that modify the meadian fragilites to account for flow rates and debris; first row is without large debris and the second one is with considering large debris
			# the factors above are for components that are vulnerable to high flows and debris

	def preallocate_space(self):
		RT_size = len(self.RT)
		time_size = len(self.time_inst) + 1

		self.pole_frag_med_tsu = np.zeros(self.n_nodes*4).reshape(self.n_nodes, 4)	#note: drs, look into if "4" can be hardcoded. 
		self.pole_frag_covm_tsu = np.zeros(4*4*self.n_nodes).reshape(4, 4, self.n_nodes)

		self.wtp_conn = np.ones(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)
		self.wps_conn = np.ones(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims*len(self.wps_nodes)).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims, len(self.wps_nodes))
		self.wwtp_conn = np.ones(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)
		self.lift_stn_conn = np.ones(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims*len(self.lift_st_nodes)).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims, len(self.lift_st_nodes))

		self.time_wtp_conn = 20000*np.ones(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)	# multuplying with a large number for comparison later
		self.time_wps_conn = 20000*np.ones(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims*len(self.wps_nodes)).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims, len(self.wps_nodes))
		self.time_wwtp_conn = 20000*np.ones(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)	# multuplying with a large number for comparison later
		self.time_lift_stn_conn = 20000*np.ones(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims*len(self.lift_st_nodes)).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims, len(self.lift_st_nodes))

		self.time_to_90p_conn_SS = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)
		self.time_to_100p_conn_SS = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)

		self.frac_closed_links = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims*time_size).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims, time_size)
		self.frac_tax_lots_conn_SS = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims*time_size).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims, time_size)
		self.rep_cost_poles = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)

		self.rep_time_SS_gen = np.zeros(self.fast_vals*self.retrofit_vals*self.n_sims*5).reshape(self.fast_vals, self.retrofit_vals, self.n_sims, 5)	# note: drs, look into if "5" can be hardcoded. 
		self.DS_SS = np.zeros(self.fast_vals*self.retrofit_vals*self.n_sims*RT_size).reshape(self.fast_vals, self.retrofit_vals, self.n_sims, RT_size)
		self.rep_time_SS = np.zeros(self.fast_vals*self.retrofit_vals*self.n_sims*RT_size).reshape(self.fast_vals, self.retrofit_vals, self.n_sims, RT_size)
		self.rep_cost_SS = np.zeros(self.fast_vals*self.retrofit_vals*self.n_sims*RT_size).reshape(self.fast_vals, self.retrofit_vals, self.n_sims, RT_size)
		self.func_SS_tsu = np.zeros(self.fast_vals*self.retrofit_vals*self.n_sims*RT_size).reshape(self.fast_vals, self.retrofit_vals, self.n_sims, RT_size)

		self.pole_failure = np.zeros((RT_size, len(self.pole_y_n), self.n_sims))


	#  ~~ from self.run() ~~ 
	def print_runinfo(self):
		print('n_sims: {}' .format(self.n_sims))
		print('fast_vals: {}' .format(self.fast_vals))
		print('retrofit_vals: {}' .format(self.retrofit_vals))
		print('plot results: {}' .format(self.plot_tf))
		print('write results: {}\n' .format(self.write_tf))


	def define_pole_rep_variables(self):
		if self.fast == 0:
			self.pole_rep_time_mu_tsu = np.array([1, 5, 20, 90])		# mean repair time for roads for DS2-DS4
			self.pole_rep_time_std_tsu = np.array([1, 5, 20, 90])*0.5	# std for repair time
		elif self.fast == 1:
			self.pole_rep_time_mu_tsu = np.array([1, 5, 20, 90])*0.5	# mean repair time for roads for DS2-DS4
			self.pole_rep_time_std_tsu = np.array([1, 5, 20, 90])*0.25	# std for repair time

		self.pole_rep_time_cov_tsu = self.pole_rep_time_std_tsu/self.pole_rep_time_mu_tsu # COV of repiar time
		self.pole_rep_time_log_med_tsu = np.log(self.pole_rep_time_mu_tsu/np.sqrt(self.pole_rep_time_cov_tsu**2+1)) # lognormal parameters for repair time model
		self.pole_rep_time_beta_tsu = np.sqrt(np.log(self.pole_rep_time_cov_tsu**2+1))
		self.pole_rep_time_covm_tsu = self.pole_rep_time_beta_tsu[:,None]*self.pole_rep_time_beta_tsu

	def define_pole_frag_variables(self):
		if self.retrofit == 0:
			self.pole_frag_med_gen_tsu = np.array([2.4, 6.2, 12.0, 18.0])/3.2808 # median fragility values - pgd in inches
			self.pole_frag_beta_tsu = np.array([0.4, 0.4, 0.4, 0.4]) # dispersion around fragility values

		if self.retrofit == 1: 
			self.pole_frag_med_gen_tsu = np.array([2.4, 6.2, 12.0, 18.0])*(1.25/3.2808) # median fragility values - pgd in inches
			self.pole_frag_beta_tsu = np.array([0.4, 0.4, 0.4, 0.4]) # dispersion around fragility values
			# self.pole_frag_covm_tsu = self.pole_frag_beta_tsu[:,None]*self.pole_frag_beta_tsu*(0.75*0.75)		


	def define_IMs(self):
		self.IM_pole_d = self.node_IM_tsu[:, self.r] # units: m
		self.IM_pole_d[self.IM_pole_d<0] = 0

		self.IM_pole_v = self.node_IM_tsu[:, self.r+len(self.RT)]	# units: m
		self.IM_pole_v[self.IM_pole_v<0] = 0

	def define_flowspeeds(self):
		self.flow_speed_type = 2*np.ones(self.n_nodes) 			# moderate flow speed
		self.flow_speed_type[self.IM_pole_v<=1.0] = 1	# low flow speed
		self.flow_speed_type[self.IM_pole_v>=5.0] = 3	# high flow speed

	def redefine_pole_frag_variables(self):
		for i in range(self.n_nodes):
			if self.flow_speed_type[i] == 1:
				self.pole_frag_med_tsu[i,:] = self.pole_frag_med_gen_tsu/(self.frag_mod_fac_SS[0,0])
				self.pole_frag_beta_tsu = np.array([0.4, 0.4, 0.4, 0.4])
			elif self.flow_speed_type[i] == 2:
				self.pole_frag_med_tsu[i,:] = self.pole_frag_med_gen_tsu/(self.frag_mod_fac_SS[0,1])
				self.pole_frag_beta_tsu = np.array([0.5, 0.5, 0.5, 0.5])
			elif self.flow_speed_type[i] == 3:
				self.pole_frag_med_tsu[i,:] = self.pole_frag_med_gen_tsu/(self.frag_mod_fac_SS[0,2])
				self.pole_frag_beta_tsu = np.array([0.5, 0.5, 0.5, 0.5])

			if self.retrofit == 0:
				self.pole_frag_covm_tsu[:,:,i] = self.pole_frag_beta_tsu[:,None]*self.pole_frag_beta_tsu 	# roadway fragility covariance matrix
			elif self.retrofit == 1:
				self.pole_frag_covm_tsu[:,:,i] = (self.pole_frag_beta_tsu[:,None]*self.pole_frag_beta_tsu)*(0.75*0.75)	# roadway fragility covariance matrix	
	
	def down_poles_eval(self, DS, DS_item, per_fail):
		for i, obj in enumerate(DS):
			self.down_poles_l_tsu[DS_item == obj] = np.random.uniform(0,1,(np.sum(DS_item==obj))) <= per_fail[i]

	def conn_analysis1(self):
		np.random.seed(self.i)
		Capacity_poles_tsu, rep_time_poles_all_tsu = DCH.capacity_reptime_gen_2d(self.pole_frag_med_tsu, self.pole_frag_covm_tsu, self.pole_rep_time_log_med_tsu, self.pole_rep_time_covm_tsu, self.n_nodes)
		DS_levels = [2,3,4,5]
		per_fail = [0.02, 0.20, 0.60, 1.00]	# assuming that there is 2%, 20%, 60% and 100% chance of road closure for slight, moderate, extensive, and complete damage respectively.

		self.down_poles, DS_poles_tsu = DCH.DS_eval(self.IM_pole_d, Capacity_poles_tsu, 3)
		DS_poles_tsu[self.pole_y_n==0] = 1

		self.down_poles_l_tsu = np.zeros(self.n_nodes, dtype = bool)
		self.down_poles_eval(DS_levels, DS_poles_tsu, per_fail)

		self.down_poles_l = self.down_poles_l_tsu[:]
		self.down_poles = np.where(self.down_poles_l)[0]
		self.rep_cost_poles[self.fast, self.retrofit, self.r, self.i] = np.sum(self.damage_ratio_poles_tsu[DS_poles_tsu.astype(int)-1] * self.pole_cost)
		self.rep_time_poles = np.ceil(np.array([rep_time_poles_all_tsu[int(i), int(obj)-1] for i, obj in enumerate(DS_poles_tsu)])) #* self.down_poles_l

		# creating post-earthquake adjacency matrix - at day 0
		G_post = self.G.copy()
		if len(self.down_poles) > 0:
			G_post = DCH.delete_edge_rowcol(G_post, self.start_node, self.end_node, self.down_poles)
		self.frac_closed_links[self.fast, self.retrofit, self.r, self.i, :] = G_post.ecount()/self.n_links
		self.bins = np.array(DCH.conn_comp(G_post, self.n_nodes))
		self.pole_failure[self.r, self.down_poles, self.i] = 1

		# post-earthquake connectivity analysis
		self.SS_bin = self.bins[self.SS_node]
		# connectivity to individual water pumps
		self.wps_conn[self.fast, self.retrofit, self.r, self.i, :] = self.bins[self.wps_nodes] == self.SS_bin
		# connectivity to wtp
		self.wtp_conn[self.fast, self.retrofit, self.r, self.i] = self.bins[self.wtp_node] == self.SS_bin
		# connectivity to wwtp
		self.wwtp_conn[self.fast, self.retrofit, self.r, self.i] = self.bins[self.wwtp_node] == self.SS_bin
		# connectivity to individual lift stations
		self.lift_stn_conn[self.fast, self.retrofit, self.r, self.i, :] = self.bins[self.lift_st_nodes] == self.SS_bin 

	def conn_analysis2(self):
		self.down_poles_l[self.rep_time_poles<=self.unique_reptime_poles[self.t]] = 0
		self.down_poles = np.where(self.down_poles_l)[0]
		
		G_post = self.G.copy()
		if len(self.down_poles) > 0:
			G_post = DCH.delete_edge_rowcol(G_post, self.start_node, self.end_node, self.down_poles)
		
		self.frac_closed_links[self.fast, self.retrofit, self.r, self.i, self.unique_reptime_poles[self.t]:] = G_post.ecount()/self.n_links
		self.bins = DCH.conn_comp(G_post, self.n_nodes)
		self.SS_bin = self.bins[self.SS_node]

	def time_eval(self):
		self.time_wps_conn[self.fast, self.retrofit, self.r, self.i,:] = np.amin([np.squeeze(self.time_wps_conn[self.fast, self.retrofit, self.r,self.i,:]), np.zeros(len(self.wps_nodes))*(self.bins[self.wps_nodes] == self.SS_bin) + 20000*(self.bins[self.wps_nodes] != self.SS_bin)], axis=0)
		# time for connectivity to wtp
		self.time_wtp_conn[self.fast, self.retrofit, self.r, self.i] = np.amin([self.time_wtp_conn[self.fast, self.retrofit, self.r, self.i], (0*(self.bins[self.wtp_node] == self.SS_bin) + 20000*(self.bins[self.wtp_node] != self.SS_bin)) ], axis=0)
		# time for connectivity to wwtp
		self.time_wwtp_conn[self.fast, self.retrofit, self.r, self.i] = np.amin([self.time_wwtp_conn[self.fast, self.retrofit, self.r, self.i], (0*(self.bins[self.wwtp_node] == self.SS_bin) + 20000*(self.bins[self.wwtp_node] != self.SS_bin)) ], axis=0)
		# time for connectivity to individual lift stattions
		self.time_lift_stn_conn[self.fast, self.retrofit, self.r, self.i] = np.amin([np.squeeze(self.time_lift_stn_conn[self.fast, self.retrofit, self.r,self.i,:]), np.zeros(len(self.lift_st_nodes))*(self.bins[self.lift_st_nodes] == self.SS_bin) + 20000*(self.bins[self.lift_st_nodes] != self.SS_bin)], axis=0)

	def substation_conn(self, init_loop = False):
		tax_lots_conn_SS = self.bins[self.tax_lot_node_ID] == self.SS_bin # tax lots connected to the SS
		if init_loop == True:
			self.frac_tax_lots_conn_SS[self.fast, self.retrofit, self.r, self.i, :] = np.sum(tax_lots_conn_SS*(self.tax_lot_type>0))/np.sum(self.tax_lot_type>0)
		
		elif init_loop == False:
			self.frac_tax_lots_conn_SS[self.fast, self.retrofit, self.r, self.i, self.unique_reptime_poles[self.t]:] = np.sum(tax_lots_conn_SS*(self.tax_lot_type>0))/np.sum(self.tax_lot_type>0)
			if (self.frac_tax_lots_conn_SS[self.fast, self.retrofit, self.r, self.i,  self.unique_reptime_poles[self.t]] > 0.9) & (self.frac_tax_lots_conn_SS[self.fast, self.retrofit, self.r, self.i,  self.unique_reptime_poles[self.t]-1] < 0.9):
				self.time_to_90p_conn_SS[self.fast, self.retrofit, self.r, self.i] = self.unique_reptime_poles[self.t]
			if (self.frac_tax_lots_conn_SS[self.fast, self.retrofit, self.r, self.i,  self.unique_reptime_poles[self.t]] == 1) & (self.frac_tax_lots_conn_SS[self.fast, self.retrofit, self.r, self.i,  self.unique_reptime_poles[self.t]-1] < 1):
				self.time_to_100p_conn_SS[self.fast, self.retrofit, self.r, self.i] = self.unique_reptime_poles[self.t]

	def complete_loop(self):
		self.frac_tax_lots_conn_SS[self.fast, self.retrofit, self.r, self.i, self.unique_reptime_poles[self.t]+1:] = 1
		self.frac_closed_links[self.fast, self.retrofit, self.r, self.i, self.unique_reptime_poles[self.t]+1:] = 0

	def time_to_p_conn_eval(self):
		if self.frac_tax_lots_conn_SS[self.fast, self.retrofit, self.r, self.i][-1] < 0.9:
			self.time_to_90p_conn_SS[self.fast, self.retrofit, self.r, self.i] = 2000		# 2000 is just any number greater than 365; to show that it takes longer than 365 days
			self.time_to_100p_conn_SS[self.fast, self.retrofit, self.r, self.i] = 2000
		elif self.frac_tax_lots_conn_SS[self.fast, self.retrofit, self.r, self.i][-1] < 1.0:
			self.time_to_100p_conn_SS[self.fast, self.retrofit, self.r, self.i] = 2000

	def plot_results(self):
		plt.figure()
		for rt in range(len(self.RT)):
			temp_avg = np.mean(self.frac_tax_lots_conn_SS[self.fast, self.retrofit, rt], axis=0)
			plt.plot(temp_avg, linewidth = 0.75, label= str(rt))
		plt.legend()
		plt.grid()

	def define_SS_IMs(self):
		self.IM_SS_d = np.array([0.041942, 0.042102, 0.063278, 0.311791, 1.86667, 3.51272, 5.46889, 6.79045])
		self.IM_SS_v = np.array([0.014642, 0.014649, 0.029406, 0.145544, 1.57545, 3.57164, 4.65251, 5.19284])
		self.damage_ratio_SS_tsu = np.array([0.0, 0.02, 0.10, 0.50, 1.00])

	def define_SS_frag_variables(self):
		self.SS_frag_med_tsu = np.array([2.6, 4.8, 8.4, 13.1])/(3.2808*self.frag_mod_fac_SS[0,self.flow_speed_type-1])	# fragility for water treatement plants w/ un-anchored components
		self.SS_frag_beta_tsu = np.array([0.4, 0.4, 0.4, 0.4]) + 0.1*(self.IM_SS_v[self.r]>1)
		self.SS_frag_covm_tsu = self.SS_frag_beta_tsu[:,None]*self.SS_frag_beta_tsu


	def define_SS_rep_variables(self):
		if self.fast == 0:
			self.SS_rep_time_mu_tsu = np.array([1, 5, 20, 120]) # mean repair time for water treatement plants for DS2-DS5
			self.SS_rep_time_std_tsu = np.array([1, 5, 20, 120])*0.5 # std for repair time		
		else:
			self.SS_rep_time_mu_tsu = np.array([1, 5, 20, 120])*0.5 # mean repair time for water treatement plants for DS2-DS5
			self.SS_rep_time_std_tsu = np.array([1, 5, 20, 120])*0.25 # std for repair time

		self.SS_rep_time_cov_tsu = self.SS_rep_time_std_tsu/self.SS_rep_time_mu_tsu # COV of repiar time
		self.SS_rep_time_log_med_tsu = np.log(self.SS_rep_time_mu_tsu/np.sqrt(self.SS_rep_time_cov_tsu**2+1)) # lognormal parameters for repair time model
		self.SS_rep_time_beta_tsu = np.sqrt(np.log(self.SS_rep_time_cov_tsu**2+1))
		self.SS_rep_time_covm_tsu = self.SS_rep_time_beta_tsu[:,None]*self.SS_rep_time_beta_tsu

		self.rep_time_SS_gen[self.fast, self.retrofit] = np.column_stack((np.zeros(self.n_sims), np.exp(np.random.multivariate_normal(self.SS_rep_time_log_med_tsu, self.SS_rep_time_covm_tsu, self.n_sims))))

	def define_flowspeeds_SS(self):
		self.flow_speed_type = 2	#*np.ones(self.n_nodes) 			# moderate flow speed. note: drs, unsure why this is single value in matlab
		if self.IM_SS_v[self.r] <= 1: self.flow_speed_type = 1
		if self.IM_SS_v[self.r] >= 5: self.flow_speed_type = 3


	def wtp_eval(self):
		np.random.seed(1337)
		Capacity_SS_tsu, _ = DCH.capacity_reptime_gen(self.SS_frag_med_tsu, self.SS_frag_covm_tsu, self.SS_rep_time_log_med_tsu, self.SS_rep_time_covm_tsu, self.n_sims)	# generating correlated capacity estimates for wtp
		_, self.DS_SS[self.fast, self.retrofit, :, self.r] = DCH.DS_eval_1d(self.IM_SS_d[self.r], Capacity_SS_tsu, 999)

		self.rep_time_SS[self.fast, self.retrofit, :, self.r] = np.array([self.rep_time_SS_gen[self.fast, self.retrofit, int(i), int(obj)-1] for i, obj in enumerate(self.DS_SS[self.fast, self.retrofit, :, self.r])])
		self.rep_cost_SS[self.fast, self.retrofit, :, self.r] = self.substation_cost*self.damage_ratio_SS_tsu[self.DS_SS[self.fast, self.retrofit, :, self.r].astype(int)-1]
		
		self.func_SS_tsu[self.fast, self.retrofit, self.DS_SS[self.fast, self.retrofit, :, self.r]==1, self.r] = 1
		self.func_SS_tsu[self.fast, self.retrofit, self.DS_SS[self.fast, self.retrofit, :, self.r]==2, self.r] = np.logical_not(np.random.uniform(0,1,(np.sum(self.DS_SS[self.fast, self.retrofit, :, self.r] == 2))) <= 0.02)
		self.func_SS_tsu[self.fast, self.retrofit, self.DS_SS[self.fast, self.retrofit, :, self.r]==3, self.r] = np.logical_not(np.random.uniform(0,1,(np.sum(self.DS_SS[self.fast, self.retrofit, :, self.r] == 3))) <= 0.20)
		self.func_SS_tsu[self.fast, self.retrofit, self.DS_SS[self.fast, self.retrofit, :, self.r]==4, self.r] = np.logical_not(np.random.uniform(0,1,(np.sum(self.DS_SS[self.fast, self.retrofit, :, self.r] == 4))) <= 0.60)
		self.func_SS_tsu[self.fast, self.retrofit, self.DS_SS[self.fast, self.retrofit, :, self.r]==5, self.r] = np.logical_not(np.random.uniform(0,1,(np.sum(self.DS_SS[self.fast, self.retrofit, :, self.r] == 5))) <= 1.00)


	# ~~~ tertiary and beyond methods ~~~
	def readmat(self, matfile, key, dtype, idx_0 = False):
		"""
		matfile = matfile to be read
		key = key in matfile to be read
		dtype = data type. includes:
			-'none': no conversion
			-'array': numpy array
			-'array_flat': flattened numpy array
			-'adj_list': adjacency matrix to list
		idx_0 = alrady 0 indexed (True/False). If True, 1 is subtractd from all values
		"""
		var = io.loadmat(matfile)[key]
		if dtype == 'none':
			pass
		elif dtype == 'array':
			var = np.array(var)
		elif dtype == 'array_flat':
			var = np.array(var).flatten()
		elif dtype == 'adj_list':
			var = (np.array(var.todense()) > 0).tolist()

		if idx_0 == True:
			var -= 1

		return var

	def misc_data_conversions(self):
		self.tax_lot_link_ID = np.array(self.tax_lot_info_raw[:,38]).astype(int).flatten() - 1	# roadway link closest to the tax lot
		self.tax_lot_type = np.array(self.tax_lot_info_raw[:,4])									# building type; 0 means no building
		self.tax_lot_pump_region = np.array(self.tax_lot_info_raw[:,40] - 1) 			# information on the pump (region) that serves the tax lot
		self.tax_lot_node_ID = np.array(self.tax_lot_info_raw[:,41] - 1).astype(int)				# information on the pump (region) that serves the tax lot
		
		tax_lot_pole_ID = self.tax_lot_info_raw[:,43] - 1 				# information on the pump (region) that serves the tax lot
		is_pole = self.tax_lot_info_raw[:,42]							# to check if the node is actually a pole
		temp = np.where(is_pole == 0)[0]
		for i in temp:
			ind = np.where((tax_lot_pole_ID[i]==self.pole_ID) & (self.pole_y_n == 1))[0]
			self.tax_lot_node_ID[i] = ind

edt = EPN_damage_tsu()
if edt.__dict__['write_tf'] == True:
	h5wrt(edt)

plt.show()


