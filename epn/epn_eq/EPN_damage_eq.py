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

class EPN_damage_eq():
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
					timer = time.time()

					for self.i in range(self.n_sims):
						self.conn_analysis1()
						self.time_eval()
						self.substation_conn(init_loop = True)
					
						self.unique_reptime_poles = np.unique(self.rep_time_poles[self.rep_time_poles != 0]).astype(int)
						for self.t in range(len(self.unique_reptime_poles)):
							self.conn_analysis2()
							self.time_eval()
							self.substation_conn(init_loop = False)


							if  np.sum(self.down_poles_1) == 0:
								self.complete_loop()
								break
					
						self.time_to_p_conn_eval()
					print('elapsed time: {}' .format(time.time() - timer))

				if self.plot_tf == True:
					self.plot_results()

		# ~~~ damage to substation ~~~
		self.define_SS_IMs()
		for self.fast in range(self.fast_vals):
			for self.retrofit in range(self.retrofit_vals):
				self.define_SS_frag_variables()
				self.define_SS_rep_variables()
			
				for self.r in range(len(self.RT)): 		#loop through return intervals
					self.wtp_eval()


	# ~~~ secondary methods ~~~
	#  ~~ from self.setup() ~~
	def load_mat_files(self):
		self.node_IM = self.readmat('node_IM_eq.mat', 'node_IM', dtype = 'array')

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

		self.damage_ratio_poles_eq = np.array([0.0, 0.05, 0.15, 0.6, 1.0]) 
		self.frag_mod_fac_SS = np.array([[1.0, 1.5, 2.0], [1.5, 2.5, 3.0]])	# factors that modify the meadian fragilites to account for flow rates and debris; first row is without large debris and the second one is with considering large debris
			# the factors above are for components that are vulnerable to high flows and debris
		self.outfilename = 'epn_damage_eq'

	def preallocate_space(self):
		RT_size = len(self.RT)
		time_size = len(self.time_inst) + 1

		self.wtp_conn = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)
		self.wps_conn = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims*len(self.wps_nodes)).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims, len(self.wps_nodes))
		self.wwtp_conn = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)
		self.lift_stn_conn = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims*len(self.lift_st_nodes)).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims, len(self.lift_st_nodes))

		self.time_wtp_conn = 20000*np.ones(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)	# multuplying with a large number for comparison later
		self.time_wps_conn = 20000*np.ones(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims*len(self.wps_nodes)).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims, len(self.wps_nodes))
		self.time_wwtp_conn = 20000*np.ones(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)	# multuplying with a large number for comparison later
		self.time_lift_stn_conn = 20000*np.ones(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims*len(self.lift_st_nodes)).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims, len(self.lift_st_nodes))

		self.time_to_90p_conn_SS = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)
		self.time_to_100p_conn_SS = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)

		self.frac_closed_links = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims*time_size).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims, time_size)
		self.frac_tax_lots_conn_SS = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims*time_size).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims, time_size)
		self.rep_cost_poles = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)

		self.rep_time_SS_gen = np.zeros(self.fast_vals*self.retrofit_vals*self.n_sims*5).reshape(self.fast_vals, self.retrofit_vals, self.n_sims, 5)	# note: drs, "5" may be an issue. look into generalizing
		self.DS_SS = np.zeros(self.fast_vals*self.retrofit_vals*self.n_sims*RT_size).reshape(self.fast_vals, self.retrofit_vals, self.n_sims, RT_size)
		self.rep_time_SS = np.zeros(self.fast_vals*self.retrofit_vals*self.n_sims*RT_size).reshape(self.fast_vals, self.retrofit_vals, self.n_sims, RT_size)
		self.rep_cost_SS = np.zeros(self.fast_vals*self.retrofit_vals*self.n_sims*RT_size).reshape(self.fast_vals, self.retrofit_vals, self.n_sims, RT_size)
		self.func_SS = np.zeros(self.fast_vals*self.retrofit_vals*self.n_sims*RT_size).reshape(self.fast_vals, self.retrofit_vals, self.n_sims, RT_size)

		self.link_failure = np.zeros((RT_size, self.n_links, self.n_sims))
		self.pole_failure = np.zeros((RT_size, len(self.pole_y_n), self.n_sims))


	#  ~~ from self.run() ~~ 
	def print_runinfo(self):
		print('n_sims: {}' .format(self.n_sims))
		print('fast_vals: {}' .format(self.fast_vals))
		print('retrofit_vals: {}' .format(self.retrofit_vals))
		print('plot results: {}' .format(self.plot_tf))
		print('write results: {}\n' .format(self.write_tf))


	def define_pole_rep_variables(self):
		self.pole_rep_time_mu_eq = np.array([0.3, 1.0, 3.0, 7.0]) # mean repair time for roads for DS2-DS4
		self.pole_rep_time_std_eq = np.array([0.2, 0.5, 1.5, 3.0]) # std for repair time       

		if self.fast == 1:
			self.pole_rep_time_mu_eq = np.array([0.3, 1.0, 3.0, 7.0])*0.5 # mean repair time for roads for DS2-DS4
			self.pole_rep_time_std_eq = np.array([0.2, 0.5, 1.5, 3.0])*0.5 # std for repair time       

		self.pole_rep_time_cov_eq = self.pole_rep_time_std_eq/self.pole_rep_time_mu_eq # COV of repiar time
		self.pole_rep_time_log_med_eq = np.log(self.pole_rep_time_mu_eq/np.sqrt(self.pole_rep_time_cov_eq**2+1)) # lognormal parameters for repair time model
		self.pole_rep_time_beta_eq = np.sqrt(np.log(self.pole_rep_time_cov_eq**2+1))
		self.pole_rep_time_covm_eq = self.pole_rep_time_beta_eq[:,None]*self.pole_rep_time_beta_eq


	def define_pole_frag_variables(self):
		self.pole_frag_med_eq = np.array([0.24, 0.33, 0.58, 0.89]) # median fragility values - pgd in inches
		self.pole_frag_beta_eq = np.array([0.25, 0.20, 0.15, 0.15]) # dispersion around fragility values

		if self.retrofit == 1: 
			self.pole_frag_med_eq = np.array([0.28, 0.4, 0.74, 1.10]) # median fragility values - pgd in inches
			self.pole_frag_beta_eq = np.array([0.3, 0.2, 0.15, 0.15]) # dispersion around fragility values

		self.pole_frag_covm = self.pole_frag_beta_eq[:,None]*self.pole_frag_beta_eq	# roadway fragility covariance matrix		


	def define_IMs(self):
		self.IM_pole_pga = self.node_IM[:,self.r] # units: g
		self.IM_pole_pga[self.IM_pole_pga<0] = np.mean(self.IM_pole_pga[self.IM_pole_pga>0])

	def conn_analysis1(self):
		np.random.seed(self.i)
		Capacity_poles_eq, rep_time_poles_all_eq = DCH.capacity_reptime_gen(self.pole_frag_med_eq, self.pole_frag_covm, self.pole_rep_time_log_med_eq, self.pole_rep_time_covm_eq, self.n_nodes)

		self.down_poles_1, DS_poles_EQ = DCH.DS_eval(self.IM_pole_pga, Capacity_poles_eq, 3)
		DS_poles_EQ[self.pole_y_n==0] = 1
		self.down_poles = np.where(self.down_poles_1)[0]

		self.rep_cost_poles[self.fast, self.retrofit, self.r, self.i] = np.sum(self.damage_ratio_poles_eq[DS_poles_EQ.astype(int)-1] * self.pole_cost)
		self.rep_time_poles = np.ceil(np.array([rep_time_poles_all_eq[int(i), int(obj)-1] for i, obj in enumerate(DS_poles_EQ)])) * self.down_poles_1

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
		self.down_poles_1[self.rep_time_poles<=self.unique_reptime_poles[self.t]] = 0
		self.down_poles = np.where(self.down_poles_1)[0]
		
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
		self.IM_SS = np.array([0.101, 0.386, 0.451, 0.636, 0.751, 0.941, 0.996, 1.25])
		self.damage_ratio_SS_eq = np.array([0.0, 0.05, 0.11, 0.55, 1.00])

	def define_SS_frag_variables(self):
		if self.retrofit == 0:	#un-anchored and un elevated
			self.SS_frag_med_eq = np.array([0.13, 0.26, 0.34, 0.74]) # fragility for water treatement plants w/ un-anchored components
			self.SS_frag_beta_eq = np.array([0.65, 0.50, 0.40, 0.40])
		elif self.retrofit == 1: # anchored
			self.SS_frag_med_eq = np.array([0.15, 0.29, 0.45, 0.90]) # fragility for water treatement plants w/ un-anchored components
			self.SS_frag_beta_eq = np.array([0.70, 0.55, 0.45, 0.45])

		self.SS_frag_covm_eq = self.SS_frag_beta_eq[:,None]*self.SS_frag_beta_eq

	def define_SS_rep_variables(self):
		if self.fast == 0:
			self.SS_rep_time_mu_eq = np.array([1, 3, 7, 30]) # mean repair time for water treatement plants for DS2-DS5
			self.SS_rep_time_std_eq = np.array([0.5, 1.5, 3.5, 15.0]) # std for repair time		
		else:
			self.SS_rep_time_mu_eq = np.array([1, 3, 7, 30])*0.5 # mean repair time for water treatement plants for DS2-DS5
			self.SS_rep_time_std_eq = np.array([0.5, 1.5, 3.5, 15.0])*0.5 # std for repair time

		self.SS_rep_time_cov_eq = self.SS_rep_time_std_eq/self.SS_rep_time_mu_eq # COV of repiar time
		self.SS_rep_time_log_med_eq = np.log(self.SS_rep_time_mu_eq/np.sqrt(self.SS_rep_time_cov_eq**2+1)) # lognormal parameters for repair time model
		self.SS_rep_time_beta_eq = np.sqrt(np.log(self.SS_rep_time_cov_eq**2+1))
		self.SS_rep_time_covm_eq = self.SS_rep_time_beta_eq[:,None]*self.SS_rep_time_beta_eq

		a, self.rep_time_SS_gen[self.fast, self.retrofit] = DCH.capacity_reptime_gen(self.SS_frag_med_eq, self.SS_frag_covm_eq, self.SS_rep_time_log_med_eq, self.SS_rep_time_covm_eq, self.n_sims)

	def wtp_eval(self):
		np.random.seed(1337)
		Capacity_SS_eq, _ = DCH.capacity_reptime_gen(self.SS_frag_med_eq, self.SS_frag_covm_eq, self.SS_rep_time_log_med_eq, self.SS_rep_time_covm_eq, self.n_sims)	# generating correlated capacity estimates for wtp
		_, self.DS_SS[self.fast, self.retrofit, :, self.r] = DCH.DS_eval_1d(self.IM_SS[self.r], Capacity_SS_eq, 999)

		self.rep_time_SS[self.fast, self.retrofit, :, self.r] = np.array([self.rep_time_SS_gen[self.fast, self.retrofit, int(i), int(obj)-1] for i, obj in enumerate(self.DS_SS[self.fast, self.retrofit, :, self.r])])
		self.rep_cost_SS[self.fast, self.retrofit, :, self.r] = self.substation_cost*self.damage_ratio_SS_eq[self.DS_SS[self.fast, self.retrofit, :, self.r].astype(int)-1]
		self.func_SS[self.fast, self.retrofit, :, self.r] = self.DS_SS[self.fast, self.retrofit, :, self.r] <= 3



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


ede = EPN_damage_eq()

if ede.__dict__['write_tf'] == True:
	h5wrt(ede)

plt.show()


