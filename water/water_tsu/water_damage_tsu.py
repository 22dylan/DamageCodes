"""
% this code performs analysis of the water network in Seaside subjected to
% earthquakes
% The main objective is to determine the connectivty of the tax lot
% locations to the water supply pump locations

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
import matplotlib.pyplot as plt
import time
np.random.seed(1337)




class water_damage_tsu():
	def __init__(self):
		self.setup()
		self.run()

	# ~~~ user input up top ~~~
	def user_input_vals(self):
		self.n_sims = 10000			# number of simulations
		self.fast_vals = 1			# originally set to 2
		self.retrofit_vals = 1		# originally set to 2

		self.plot_tf = False
		self.write_tf = False

	def setup(self):
		self.user_input_vals()
		self.load_mat_files()
		self.initialize_variables()
		self.preallocate_space()
		self.G = DCH.create_graph(self.adj_water)


	def run(self):
		self.print_runinfo()
		for self.fast in range(self.fast_vals):
			print('fast: {}' .format(self.fast))
			self.define_workers()

			for self.r in range(len(self.RT)): 		#loop through return intervals
				self.define_IMs()
				self.define_flowspeeds()
				self.define_frag_vals()

				timer = time.time()
				for self.i in range(self.n_sims):
					self.define_wp_unit_DS()
					self.define_leak_break()
					self.conn_analysis1()
					self.time_eval()

					for self.t in range(len(self.unique_rep_times_pipes)):
						self.conn_analysis2()
						self.substation_conn(init_loop = False)

						if  np.sum(self.closed_pipes_ID) == 0:
							self.complete_loop()
							break

					self.time_to_p_conn_eval()

				print('elapsed time: {}' .format(time.time() - timer))

			if self.plot_tf == True:
				self.plot_results()

		# ~~~ damage to water treatment plants ~~~
		self.define_wtp_damage_ratio()
		for self.fast in range(self.fast_vals):
			self.define_wtp_rep_variables()

			for self.retrofit in range(self.retrofit_vals):
				self.rep_time_wtp_gen[self.fast, self.retrofit] = np.column_stack((np.zeros(self.n_sims), np.exp(np.random.multivariate_normal(self.wtp_rep_time_log_med, self.wtp_rep_time_covm, self.n_sims))))
				for self.r in range(len(self.RT)): 		#loop through return intervals
					self.define_wtp_frag_variables()
			
					self.wtp_eval()

		# ~~~ damage to water pumping station ~~~
		self.define_wps_damage_ratio()
		for self.fast in range(self.fast_vals):
			self.define_wps_rep_variables()

			for self.retrofit in range(self.retrofit_vals):
				self.rep_time_wps_gen[self.fast, self.retrofit] = np.column_stack((np.zeros(self.n_sims), np.exp(np.random.multivariate_normal(self.wps_rep_time_log_med, self.wps_rep_time_covm, self.n_sims))))
				for self.r in range(len(self.RT)):
					for self.np in range(self.n_wps):
						self.define_wps_frag_variables()						

						self.wps_eval()
	
		self.reorg_gis_data()		# note: drs, added this

	# ~~~ secondary methods ~~~
	#  ~~ from self.setup() ~~
	def load_mat_files(self):
		self.link_IMs = DCH.readmat('link_IM.mat', 'link_IMs', dtype = 'array')
		self.wtp_IM = DCH.readmat('link_IM.mat', 'wtp_IM', dtype = 'array')
		self.ps_IM = DCH.readmat('link_IM.mat', 'ps_IM', dtype = 'array')

		self.start_node = DCH.readmat('water_adjacency.mat', 'start_node', dtype = 'array_flat', idx_0 = True)
		self.end_node = DCH.readmat('water_adjacency.mat', 'end_node', dtype = 'array_flat', idx_0 = True)
		self.adj_water = DCH.readmat('water_adjacency.mat', 'adj_water', dtype = 'adj_list')
		self.ductile = DCH.readmat('water_adjacency.mat', 'ductile', dtype = 'array_flat')
		self.pipe_dia = DCH.readmat('water_adjacency.mat', 'pipe_dia', dtype = 'array_flat')
		self.pipe_length = DCH.readmat('water_adjacency.mat', 'pipe_length', dtype = 'array_flat')
		self.pump_num = DCH.readmat('water_adjacency.mat', 'pump_num', dtype = 'array_flat')
		self.segment = DCH.readmat('water_adjacency.mat', 'segment', dtype = 'array_flat')
		
		self.tax_lot_info_raw = DCH.readmat('tax_lot_info.mat', 'tax_lot_info', dtype = 'array')

		self.misc_data_conversions()

	def initialize_variables(self):
		self.RT = np.array([100, 200, 250, 500 ,1000, 2500, 5000, 10000]) # return period of earthquake events (in years)
		self.time_inst = np.array(range(3*365))+1		# time instances (or, in this case, # days in 3 years)
		self.g = 386.4									# intensity measure values in g
		self.n_wps = 3

		"""
		frag_mod_fac1 and 2 modify the meadian fragilites to account for flow rates and debris; first row is without large debris and the second one is with considering large debris
			frag_mod_fac1 - the factors above are for components that are vulnerable to high flows but not to debris
			frag_mod_fac2 - the factors above are for components that are vulnerable to high flows and debris
		"""
		self.frag_mod_fac1 = np.array([[1.0, 1.4, 1.6], [1.1, 1.5, 1.8]])					
		self.frag_mod_fac2 = np.array([[1.0, 1.5, 2.0], [1.5, 2.5, 3.0]])	

		self.n_nodes = np.shape(self.adj_water)[0]
		self.n_links = len(self.start_node)
		self.link_ID = np.array(range(0,self.n_links))

		# node numbers of water pumps in the water network
		self.wpump_node = np.array([0, 415, 364])	# pump 1,2 and 3 respectively

		self.tax_lot_start_node = self.start_node[self.tax_lot_link_ID]
		self.tax_lot_end_node = self.end_node[self.tax_lot_link_ID]
		
		# all fragility estimates based on HAZUS
		self.tot_len_ductile = np.sum(self.ductile*self.pipe_length)/3280.84 # total length of brittle pipes (in km)
		self.tot_len_brittle = np.sum(self.brittle*self.pipe_length)/3280.84 # total length of ductile pipes (in km)

		# repair time parameters for roadway links
		self.wpipe_brep_rate = np.array([0.33, 0.5]) # rate of fixing breaks for pipes of diameter greater than 20" and dia less than 20" respectively -- # of breaks fixed per day per worker - assuming 16 hour shifts
		self.wpipe_lrep_rate = 2*self.wpipe_brep_rate # rate of fixing leaks for pipes of diameter greater than 20" and dia less than 20" respectively -- # of leaks fixed per day per worker - assuming 16 hour shifts

		self.pipe_unit_length = 20 # assuming 20' length for pipes in the network
		self.n_pipe_unit = np.ceil(self.pipe_length/self.pipe_unit_length) # number of pipe units per link

		self.ductile_pipe_units_start = 1+np.cumsum(self.n_pipe_unit*self.ductile)
		self.ductile_pipe_units_start = np.insert(self.ductile_pipe_units_start, 0, 1)[:-1]
		self.ductile_pipe_units_end = np.cumsum(self.n_pipe_unit*self.ductile)

		self.brittle_pipe_units_start = 1+np.cumsum(self.n_pipe_unit*self.brittle)
		self.brittle_pipe_units_start = np.insert(self.brittle_pipe_units_start, 0, 1)[:-1]
		self.brittle_pipe_units_end = np.cumsum(self.n_pipe_unit*self.brittle)

		self.mod_frag = True # modified fragility 
		
		self.outfilename = 'water_damage_tsu'
	def preallocate_space(self):
		RT_size = len(self.RT)
		time_size = len(self.time_inst) + 1
		n_damage_states = 4
		
		self.frac_closed_wpipes = np.zeros(self.fast_vals*RT_size*self.n_sims*time_size).reshape(self.fast_vals, RT_size, self.n_sims, time_size)
		self.frac_closed_wpipes1 = np.zeros(self.fast_vals*RT_size*self.n_sims*time_size).reshape(self.fast_vals, RT_size, self.n_sims, time_size)
		self.frac_closed_wpipes2 = np.zeros(self.fast_vals*RT_size*self.n_sims*time_size).reshape(self.fast_vals, RT_size, self.n_sims, time_size)
		self.frac_closed_wpipes3 = np.zeros(self.fast_vals*RT_size*self.n_sims*time_size).reshape(self.fast_vals, RT_size, self.n_sims, time_size)

		self.frac_tax_lots_conn_wpump = np.zeros(self.fast_vals*RT_size*self.n_sims*time_size).reshape(self.fast_vals, RT_size, self.n_sims, time_size)
		self.frac_tax_lots_conn_wpump1 = np.zeros(self.fast_vals*RT_size*self.n_sims*time_size).reshape(self.fast_vals, RT_size, self.n_sims, time_size)
		self.frac_tax_lots_conn_wpump2 = np.zeros(self.fast_vals*RT_size*self.n_sims*time_size).reshape(self.fast_vals, RT_size, self.n_sims, time_size)
		self.frac_tax_lots_conn_wpump3 = np.zeros(self.fast_vals*RT_size*self.n_sims*time_size).reshape(self.fast_vals, RT_size, self.n_sims, time_size)

		self.time_to_90p_conn_wpump = np.zeros(self.fast_vals*RT_size*self.n_sims).reshape(self.fast_vals, RT_size, self.n_sims)
		self.time_to_100p_conn_wpump = np.zeros(self.fast_vals*RT_size*self.n_sims).reshape(self.fast_vals, RT_size, self.n_sims)

		self.time_to_90p_conn_wpump1 = np.zeros(self.fast_vals*RT_size*self.n_sims).reshape(self.fast_vals, RT_size, self.n_sims)
		self.time_to_100p_conn_wpump1 = np.zeros(self.fast_vals*RT_size*self.n_sims).reshape(self.fast_vals, RT_size, self.n_sims)

		self.time_to_90p_conn_wpump2 = np.zeros(self.fast_vals*RT_size*self.n_sims).reshape(self.fast_vals, RT_size, self.n_sims)
		self.time_to_100p_conn_wpump2 = np.zeros(self.fast_vals*RT_size*self.n_sims).reshape(self.fast_vals, RT_size, self.n_sims)

		self.time_to_90p_conn_wpump3 = np.zeros(self.fast_vals*RT_size*self.n_sims).reshape(self.fast_vals, RT_size, self.n_sims)
		self.time_to_100p_conn_wpump3 = np.zeros(self.fast_vals*RT_size*self.n_sims).reshape(self.fast_vals, RT_size, self.n_sims)

		self.pipe_rep_cost = np.zeros(self.fast_vals*RT_size*self.n_sims).reshape(self.fast_vals, RT_size, self.n_sims)
		self.num_pipes_leak = np.zeros(self.fast_vals*RT_size*self.n_sims).reshape(self.fast_vals, RT_size, self.n_sims)
		self.num_pipes_break = np.zeros(self.fast_vals*RT_size*self.n_sims).reshape(self.fast_vals, RT_size, self.n_sims)

		self.wp_frag_med = np.zeros(self.n_links*n_damage_states).reshape(self.n_links, n_damage_states)
		self.wp_frag_beta = np.zeros(self.n_links*n_damage_states).reshape(self.n_links, n_damage_states)

		self.rep_time_wtp_gen = np.zeros(self.fast_vals*self.retrofit_vals*self.n_sims*5).reshape(self.fast_vals, self.retrofit_vals, self.n_sims, 5)	# note: drs, "5" may be an issue. look into generalizing
		self.DS_wtp = np.zeros(self.fast_vals*self.retrofit_vals*self.n_sims*RT_size).reshape(self.fast_vals, self.retrofit_vals, self.n_sims, RT_size)
		self.rep_time_wtp = np.zeros(self.fast_vals*self.retrofit_vals*self.n_sims*RT_size).reshape(self.fast_vals, self.retrofit_vals, self.n_sims, RT_size)
		self.rep_cost_wtp = np.zeros(self.fast_vals*self.retrofit_vals*self.n_sims*RT_size).reshape(self.fast_vals, self.retrofit_vals, self.n_sims, RT_size)

		self.rep_time_wps_gen = np.zeros(self.fast_vals*self.retrofit_vals*self.n_sims*5).reshape(self.fast_vals, self.retrofit_vals, self.n_sims, 5)	# note: drs, "5" may be an issue. look into generalizing
		self.DS_wps = np.zeros(self.n_wps*self.fast_vals*self.retrofit_vals*self.n_sims*RT_size).reshape(self.n_wps, self.fast_vals, self.retrofit_vals, self.n_sims, RT_size)
		self.rep_time_wps = np.zeros(self.n_wps*self.fast_vals*self.retrofit_vals*self.n_sims*RT_size).reshape(self.n_wps, self.fast_vals, self.retrofit_vals, self.n_sims, RT_size)	# note: drs, "5" may be an issue. look into generalizing
		self.rep_cost_wps = np.zeros(self.n_wps*self.fast_vals*self.retrofit_vals*self.n_sims*RT_size).reshape(self.n_wps, self.fast_vals, self.retrofit_vals, self.n_sims, RT_size)

		self.frac_tax_lots_conn_write = np.zeros((RT_size, len(self.tax_lot_type), self.n_sims))	# note: drs, added this

	#  ~~ from self.run() ~~
	def print_runinfo(self):
		print('n_sims: {}' .format(self.n_sims))
		print('fast_vals: {}' .format(self.fast_vals))
		print('retrofit_vals: {}' .format(self.retrofit_vals))
		print('plot results: {}' .format(self.plot_tf))
		print('write results: {}\n' .format(self.write_tf))

	def define_workers(self):
		if self.fast == 0:
			self.n_workers = 32
		elif self.fast == 1:
			self.n_workers = 64

	def define_IMs(self):
		self.IM_wpipe_d = self.link_IMs[:, self.r+1+len(self.RT)] # first column is link ID; flow depth - units: m
		self.IM_wpipe_d[self.IM_wpipe_d<0] = 0

		self.IM_wpipe_v = self.link_IMs[:, self.r+1]	# first column is link ID; flow speed - units: m/s
		self.IM_wpipe_v[self.IM_wpipe_v<0] = 0

	def define_flowspeeds(self):
		self.flow_speed_type = 2*np.ones(self.n_links)	# moderate flow speed
		self.flow_speed_type[self.IM_wpipe_v<=1.0] = 1	# low flow speed
		self.flow_speed_type[self.IM_wpipe_v>=5.0] = 3	# high flow speed

	def define_frag_vals(self):
		if self.mod_frag == True:
			self.med_pipe_frag = 2.5*np.array([3.6, 7.1, 10.0, 12.8])/3.2808
		else:
			self.med_pipe_frag = np.array([3.6, 7.1, 10.0, 12.8])/3.2808

		self.wp_frag_med[self.flow_speed_type==1,:] = np.tile(self.med_pipe_frag, (np.sum(self.flow_speed_type==1), 1))
		self.wp_frag_med[self.flow_speed_type==2,:] = np.tile(self.med_pipe_frag/self.frag_mod_fac1[0,1], (np.sum(self.flow_speed_type==2), 1))
		self.wp_frag_med[self.flow_speed_type==3,:] = np.tile(self.med_pipe_frag/self.frag_mod_fac1[0,2], (np.sum(self.flow_speed_type==3), 1))

		self.wp_frag_beta = np.zeros(np.shape(self.wp_frag_med))
		if self.mod_frag==True:
			self.wp_frag_beta[self.flow_speed_type==1,:] = np.tile(0.75*np.array([0.4, 0.4, 0.4, 0.4]), (np.sum(self.flow_speed_type==1), 1))
			self.wp_frag_beta[self.flow_speed_type>1,:] = np.tile(0.75*np.array([0.5, 0.5, 0.5, 0.5]), (np.sum(self.flow_speed_type>1), 1))
		else:
			self.wp_frag_beta[self.flow_speed_type==1,:] = np.tile([0.4, 0.4, 0.4, 0.4], (np.sum(self.flow_speed_type==1), 1))
			self.wp_frag_beta[self.flow_speed_type>1,:] = np.tile([0.5, 0.5, 0.5, 0.5], (np.sum(self.flow_speed_type>1), 1))

	def define_wp_unit_DS(self):
		wp_units_capacity = np.zeros(np.sum(self.n_pipe_unit).astype(int)*len(self.med_pipe_frag)).reshape(np.sum(self.n_pipe_unit).astype(int), len(self.med_pipe_frag))
		wp_units_demand = np.zeros(np.sum(self.n_pipe_unit).astype(int))
		self.wp_unit_ductile = np.zeros(np.sum(self.n_pipe_unit).astype(int))

		counter = 0
		for nl in range(self.n_links):
			wp_frag_covm = self.wp_frag_beta[nl,:][:,None]*self.wp_frag_beta[nl, :]
			wp_units_capacity[counter:counter+self.n_pipe_unit[nl].astype(int), :] =  np.exp(np.random.multivariate_normal(np.log(self.wp_frag_med[nl,:]), wp_frag_covm, size = self.n_pipe_unit[nl].astype(int)))	# generating correlated capacity estimates for water pipes
			wp_units_demand[counter:counter+self.n_pipe_unit[nl].astype(int)] = self.IM_wpipe_d[nl]		# generating correlated capacity estimates for water pipes
			self.wp_unit_ductile[counter:counter+self.n_pipe_unit[nl].astype(int)] = self.ductile[nl]
			counter += self.n_pipe_unit[nl].astype(int)

		self.wp_unit_brittle = self.wp_unit_ductile == 0
		_, self.wp_units_DS = DCH.DS_eval(wp_units_demand, wp_units_capacity, 999)


	def define_leak_break(self):
		# here it is assumed that extesnsive and moderate damages lead to leaks but complete damage lead to breaks  
		temp1 = np.where((self.wp_units_DS*self.wp_unit_ductile) == 2)[0]
		temp2 = np.where((self.wp_units_DS*self.wp_unit_ductile) == 3)[0]
		temp3 = np.where((self.wp_units_DS*self.wp_unit_ductile) == 4)[0]
		
		ductile_pipe_units_leak = np.concatenate((temp1[np.random.permutation(np.rint(len(temp1)*.02).astype(int))], temp2[np.random.permutation(np.rint(len(temp2)*.2).astype(int))], temp3[np.random.permutation(np.rint(len(temp3)*.6).astype(int))]))	#identifying the ductile pipe units with leaks
		ductile_pipe_units_break = np.where((self.wp_units_DS*self.wp_unit_ductile) > 4)[0]	# identifying the ductile pipe units with breaks

		temp1 = np.where((self.wp_units_DS*self.wp_unit_brittle) == 2)[0]
		temp2 = np.where((self.wp_units_DS*self.wp_unit_brittle) == 3)[0]
		temp3 = np.where((self.wp_units_DS*self.wp_unit_brittle) == 4)[0]
		brittle_pipe_units_leak = np.concatenate((temp1[np.random.permutation(np.rint(len(temp1)*.02).astype(int))], temp2[np.random.permutation(np.rint(len(temp2)*.2).astype(int))], temp3[np.random.permutation(np.rint(len(temp3)*.6).astype(int))]))	#identifying the ductile pipe units with leaks
		brittle_pipe_units_break = np.where((self.wp_units_DS*self.wp_unit_brittle) > 4)[0]	# identifying the ductile pipe units with breaks

		if len(ductile_pipe_units_leak) == 0:
			ductile_pipe_units_leak = np.array([0])
		if len(ductile_pipe_units_break) == 0:
			ductile_pipe_units_break = np.array([0])
		if len(brittle_pipe_units_leak) == 0:
			brittle_pipe_units_leak = np.array([0])
		if len(brittle_pipe_units_break) == 0:
			brittle_pipe_units_break = np.array([0])
		
		self.ductile_pipes_leaks = np.sum(np.logical_and([i>=self.ductile_pipe_units_start for i in ductile_pipe_units_leak], [i<=self.ductile_pipe_units_end for i in ductile_pipe_units_leak]), axis=0)		# identify the ductile links with leaks and the number of leaks on that link
		self.ductile_pipes_breaks = np.sum(np.logical_and([i>=self.ductile_pipe_units_start for i in ductile_pipe_units_break], [i<=self.ductile_pipe_units_end for i in ductile_pipe_units_break]), axis = 0)	# identify the ductile links with breaks and the number of breaks on that link
		self.brittle_pipes_leaks = np.sum(np.logical_and([i>=self.brittle_pipe_units_start for i in brittle_pipe_units_leak], [i<=self.brittle_pipe_units_end for i in brittle_pipe_units_leak]), axis = 0)		# identify the brittle links with leaks and the number of leaks on that link
		self.brittle_pipes_breaks = np.sum(np.logical_and([i>=self.brittle_pipe_units_start for i in brittle_pipe_units_break], [i<=self.brittle_pipe_units_end for i in brittle_pipe_units_break]), axis = 0)	# identify the brittle links with breaks and the number of breaks on that link
	
		self.pipes_leaks = self.ductile_pipes_leaks + self.brittle_pipes_leaks
		self.pipes_breaks = self.ductile_pipes_breaks + self.brittle_pipes_breaks

		self.num_pipes_leak[self.fast, self.r, self.i] = np.sum(self.pipes_leaks)
		self.num_pipes_break[self.fast, self.r, self.i] = np.sum(self.pipes_breaks)
		self.pipe_rep_cost[self.fast, self.r, self.i] = np.sum(self.pipes_leaks)*0.1 + np.sum(self.pipes_breaks)*0.75

	def conn_analysis1(self):
		"""
		damage state assignment assumes that 80% of the repairs are leaks
			and 20% are breaks for damage due to seismic wave propagation - as per HAZUS
			DS1 - no damage; DS2 - slight damage; DS3 - moderate damage; DS4 - complete damage
		"""
		DS_pipes = 1 + self.ductile*((self.ductile_pipes_leaks>0) + (self.ductile_pipes_breaks>0)) + self.brittle*((self.brittle_pipes_leaks>0) + (self.brittle_pipes_breaks>0))	# DS1 - no damage; DS2 - leak; DS3 - break
		DS_pipes = DS_pipes[0]		# DS_pipes was a nested array..
		self.closed_pipes = DS_pipes>1 	# assuming that the roads are closed when they have moderate or complete damage
		
		self.closed_segments = self.segment[self.closed_pipes] # pipe segments that are closed
		self.closed_segments = self.closed_segments[self.closed_segments>0]
		self.closed_pipes = self.closed_pipes + np.in1d(self.segment, self.closed_segments)

		self.frac_closed_wpipes1[self.fast, self.r, self.i, :] = np.sum(np.logical_and(self.closed_pipes, self.pump_num==1))/np.sum(self.pump_num==1)
		self.frac_closed_wpipes2[self.fast, self.r, self.i, :] = np.sum(np.logical_and(self.closed_pipes, self.pump_num==2))/np.sum(self.pump_num==2)
		self.frac_closed_wpipes3[self.fast, self.r, self.i, :] = np.sum(np.logical_and(self.closed_pipes, self.pump_num==3))/np.sum(self.pump_num==3)
		self.frac_closed_wpipes[self.fast, self.r, self.i, :] = np.mean(self.closed_pipes)

		self.closed_pipes_ID = np.where(self.closed_pipes)[0]


		# creating post-earthquake adjacency matrix - at day 0
		G_post = self.G.copy()
		if len(self.closed_pipes) > 0:
			G_post = DCH.delete_edges(G_post, self.start_node, self.end_node, self.closed_pipes_ID)
		self.bins = np.array(DCH.conn_comp(G_post, self.n_nodes))

		# post-earthquake connectivity analysis          
		self.wpump_bins = self.bins[self.wpump_node]	# there can be a maximum of three such bins  

		# connectivity to individual pumps
		self.nodes_conn_wpump1 = self.bins==self.wpump_bins[0] # nodes connected to the water pupms
		self.tax_lots_conn_wpump1 = np.logical_and(self.nodes_conn_wpump1[self.tax_lot_start_node], self.nodes_conn_wpump1[self.tax_lot_end_node]) 	# tax lots connected to the water pumps
		self.frac_tax_lots_conn_wpump1[self.fast, self.r, self.i, :] = np.sum(np.logical_and.reduce((self.tax_lots_conn_wpump1, self.logical_tax_lot_type, self.tax_lot_pump_region==0)))/np.sum(np.logical_and(self.logical_tax_lot_type, self.tax_lot_pump_region==0))	#fraction of tax lots that are buildingd and are connected to the water pumps

		self.nodes_conn_wpump2 = self.bins==self.wpump_bins[1] # nodes connected to the water pupms
		self.tax_lots_conn_wpump2 = np.logical_and(self.nodes_conn_wpump2[self.tax_lot_start_node], self.nodes_conn_wpump2[self.tax_lot_end_node]) 	# tax lots connected to the water pumps
		self.frac_tax_lots_conn_wpump2[self.fast, self.r, self.i, :] = np.sum(np.logical_and.reduce((self.tax_lots_conn_wpump2, self.logical_tax_lot_type, self.tax_lot_pump_region==1)))/np.sum(np.logical_and(self.logical_tax_lot_type, self.tax_lot_pump_region==1))	#fraction of tax lots that are buildingd and are connected to the water pumps

		self.nodes_conn_wpump3 = self.bins==self.wpump_bins[2] # nodes connected to the water pupms
		self.tax_lots_conn_wpump3 = np.logical_and(self.nodes_conn_wpump3[self.tax_lot_start_node], self.nodes_conn_wpump3[self.tax_lot_end_node]) 	# tax lots connected to the water pumps
		self.frac_tax_lots_conn_wpump3[self.fast, self.r, self.i, :] = np.sum(np.logical_and.reduce((self.tax_lots_conn_wpump3, self.logical_tax_lot_type, self.tax_lot_pump_region==2)))/np.sum(np.logical_and(self.logical_tax_lot_type, self.tax_lot_pump_region==2))	#fraction of tax lots that are buildingd and are connected to the water pumps

		# connected to any of the 3 water pumps
		self.nodes_conn_wpump = np.logical_or.reduce((self.bins==self.wpump_bins[0], self.bins==self.wpump_bins[1], self.bins==self.wpump_bins[2]))	# nodes connected to the water pumps
		self.tax_lots_conn_wpump = np.logical_and(self.nodes_conn_wpump[self.tax_lot_start_node], self.nodes_conn_wpump[self.tax_lot_end_node])		# tax lots connected to the water pumps
		self.frac_tax_lots_conn_wpump[self.fast, self.r, self.i, :] = np.sum(self.tax_lots_conn_wpump*self.logical_tax_lot_type)/np.sum(self.logical_tax_lot_type)	# fraction of tax lots that are buildingd and are connected to the water pumps  
	
		# note: drs, added this
		if (self.fast == 0):
			self.frac_tax_lots_conn_write[self.r,:, self.i] = self.tax_lots_conn_wpump*1
		# ~~~


	def conn_analysis2(self):
		self.closed_pipes[self.rep_time_pipes<=self.unique_rep_times_pipes[self.t]] = 0

		self.closed_segments = self.segment[self.closed_pipes] # pipe segments that are closed
		self.closed_segments = self.closed_segments[self.closed_segments>0]
		self.closed_pipes = self.closed_pipes + np.in1d(self.segment, self.closed_segments)

		self.frac_closed_wpipes1[self.fast, self.r, self.i, self.unique_rep_times_pipes[self.t]:] = np.sum(np.logical_and(self.closed_pipes, self.pump_num==1))/np.sum(self.pump_num==1)
		self.frac_closed_wpipes2[self.fast, self.r, self.i, self.unique_rep_times_pipes[self.t]:] = np.sum(np.logical_and(self.closed_pipes, self.pump_num==2))/np.sum(self.pump_num==2)
		self.frac_closed_wpipes3[self.fast, self.r, self.i, self.unique_rep_times_pipes[self.t]:] = np.sum(np.logical_and(self.closed_pipes, self.pump_num==3))/np.sum(self.pump_num==3)
		self.frac_closed_wpipes[self.fast, self.r, self.i, self.unique_rep_times_pipes[self.t]:] = np.mean(self.closed_pipes)	# fraction of tax lots that are buildingd and are connected to the water pumps  
	
		self.closed_pipes_ID = np.where(self.closed_pipes)[0]

		G_post = self.G.copy()
		if len(self.closed_pipes) > 0:
			G_post = DCH.delete_edges(G_post, self.start_node, self.end_node, self.closed_pipes_ID)
		self.bins = np.array(DCH.conn_comp(G_post, self.n_nodes))

		self.wpump_bins = self.bins[self.wpump_node]	# there can be a maximum of three such bins  

		# connectivity to individual pumps
		self.nodes_conn_wpump1 = self.bins==self.wpump_bins[0] # nodes connected to the water pupms
		self.tax_lots_conn_wpump1 = np.logical_and(self.nodes_conn_wpump1[self.tax_lot_start_node], self.nodes_conn_wpump1[self.tax_lot_end_node]) 	# tax lots connected to the water pumps
		self.frac_tax_lots_conn_wpump1[self.fast, self.r, self.i, self.unique_rep_times_pipes[self.t]:] = np.sum(np.logical_and.reduce((self.tax_lots_conn_wpump1, self.logical_tax_lot_type, self.tax_lot_pump_region==0)))/np.sum(np.logical_and(self.logical_tax_lot_type, self.tax_lot_pump_region==0))	#fraction of tax lots that are buildingd and are connected to the water pumps

		self.nodes_conn_wpump2 = self.bins==self.wpump_bins[1] # nodes connected to the water pupms
		self.tax_lots_conn_wpump2 = np.logical_and(self.nodes_conn_wpump2[self.tax_lot_start_node], self.nodes_conn_wpump2[self.tax_lot_end_node]) 	# tax lots connected to the water pumps
		self.frac_tax_lots_conn_wpump2[self.fast, self.r, self.i, self.unique_rep_times_pipes[self.t]:] = np.sum(np.logical_and.reduce((self.tax_lots_conn_wpump2, self.logical_tax_lot_type, self.tax_lot_pump_region==1)))/np.sum(np.logical_and(self.logical_tax_lot_type, self.tax_lot_pump_region==1))	#fraction of tax lots that are buildingd and are connected to the water pumps

		self.nodes_conn_wpump3 = self.bins==self.wpump_bins[2] # nodes connected to the water pupms
		self.tax_lots_conn_wpump3 = np.logical_and(self.nodes_conn_wpump3[self.tax_lot_start_node], self.nodes_conn_wpump3[self.tax_lot_end_node]) 	# tax lots connected to the water pumps
		self.frac_tax_lots_conn_wpump3[self.fast, self.r, self.i, self.unique_rep_times_pipes[self.t]:] = np.sum(np.logical_and.reduce((self.tax_lots_conn_wpump3, self.logical_tax_lot_type, self.tax_lot_pump_region==2)))/np.sum(np.logical_and(self.logical_tax_lot_type, self.tax_lot_pump_region==2))	#fraction of tax lots that are buildingd and are connected to the water pumps

		# connected to any of the 3 water pumps
		self.nodes_conn_wpump = np.logical_or.reduce((self.bins==self.wpump_bins[0], self.bins==self.wpump_bins[1], self.bins==self.wpump_bins[2]))	# nodes connected to the water pumps
		self.tax_lots_conn_wpump = np.logical_and(self.nodes_conn_wpump[self.tax_lot_start_node], self.nodes_conn_wpump[self.tax_lot_end_node])		# tax lots connected to the water pumps
		self.frac_tax_lots_conn_wpump[self.fast, self.r, self.i, self.unique_rep_times_pipes[self.t]:] = np.sum(self.tax_lots_conn_wpump*self.logical_tax_lot_type)/np.sum(self.logical_tax_lot_type)	# fraction of tax lots that are buildingd and are connected to the water pumps  
	
	def substation_conn(self, init_loop = False):
		if init_loop == True:
			pass
		elif init_loop == False:
			if (self.frac_tax_lots_conn_wpump[self.fast, self.r, self.i,  self.unique_rep_times_pipes[self.t]] > 0.9) & (self.frac_tax_lots_conn_wpump[self.fast, self.r, self.i,  self.unique_rep_times_pipes[self.t]-1] < 0.9):
				self.time_to_90p_conn_wpump[self.fast, self.r, self.i] = self.unique_rep_times_pipes[self.t]
			if (self.frac_tax_lots_conn_wpump[self.fast, self.r, self.i,  self.unique_rep_times_pipes[self.t]] == 1.0) & (self.frac_tax_lots_conn_wpump[self.fast, self.r, self.i,  self.unique_rep_times_pipes[self.t]-1] < 1.0):
				self.time_to_100p_conn_wpump[self.fast, self.r, self.i] = self.unique_rep_times_pipes[self.t]

			if (self.frac_tax_lots_conn_wpump1[self.fast, self.r, self.i,  self.unique_rep_times_pipes[self.t]] > 0.9) & (self.frac_tax_lots_conn_wpump1[self.fast, self.r, self.i,  self.unique_rep_times_pipes[self.t]-1] < 0.9):
				self.time_to_90p_conn_wpump1[self.fast,  self.r, self.i] = self.unique_rep_times_pipes[self.t]
			if (self.frac_tax_lots_conn_wpump1[self.fast, self.r, self.i,  self.unique_rep_times_pipes[self.t]] == 1.0) & (self.frac_tax_lots_conn_wpump1[self.fast, self.r, self.i,  self.unique_rep_times_pipes[self.t]-1] < 1.0):
				self.time_to_100p_conn_wpump1[self.fast, self.r, self.i] = self.unique_rep_times_pipes[self.t]

			if (self.frac_tax_lots_conn_wpump2[self.fast, self.r, self.i,  self.unique_rep_times_pipes[self.t]] > 0.9) & (self.frac_tax_lots_conn_wpump2[self.fast, self.r, self.i,  self.unique_rep_times_pipes[self.t]-1] < 0.9):
				self.time_to_90p_conn_wpump2[self.fast, self.r, self.i] = self.unique_rep_times_pipes[self.t]
			if (self.frac_tax_lots_conn_wpump2[self.fast, self.r, self.i,  self.unique_rep_times_pipes[self.t]] == 1.0) & (self.frac_tax_lots_conn_wpump2[self.fast, self.r, self.i,  self.unique_rep_times_pipes[self.t]-1] < 1.0):
				self.time_to_100p_conn_wpump2[self.fast, self.r, self.i] = self.unique_rep_times_pipes[self.t]

			if (self.frac_tax_lots_conn_wpump3[self.fast, self.r, self.i,  self.unique_rep_times_pipes[self.t]] > 0.9) & (self.frac_tax_lots_conn_wpump3[self.fast, self.r, self.i,  self.unique_rep_times_pipes[self.t]-1] < 0.9):
				self.time_to_90p_conn_wpump3[self.fast, self.r, self.i] = self.unique_rep_times_pipes[self.t]
			if (self.frac_tax_lots_conn_wpump3[self.fast, self.r, self.i,  self.unique_rep_times_pipes[self.t]] == 1.0) & (self.frac_tax_lots_conn_wpump3[self.fast, self.r, self.i,  self.unique_rep_times_pipes[self.t]-1] < 1.0):
				self.time_to_100p_conn_wpump3[self.fast,  self.r, self.i] = self.unique_rep_times_pipes[self.t]


	def time_eval(self):
		leak_term = np.zeros(self.n_links)
		leak_term[self.pipe_dia<20] = self.wpipe_lrep_rate[0]
		leak_term[self.pipe_dia>20] = self.wpipe_lrep_rate[1]
		break_term = np.zeros(self.n_links)
		break_term[self.pipe_dia<20] = self.wpipe_brep_rate[0]
		break_term[self.pipe_dia>20] = self.wpipe_brep_rate[1]

		# time needed to rapir all the leaks and breaks in a pipe (in number of days) based on the assumptions of number of personnel per crew; also a 16 hour shift is assumed
		rep_time_pipes_i = ((16/self.n_workers)*self.pipes_leaks/leak_term + (16/self.n_workers)*self.pipes_breaks/break_term)/24
		
		# rank the priority of links to be reparied first: larger diameter pipes get priority for repairs
		rep_rank_pipes = np.argsort(-(self.pipe_dia + 1e-3*np.random.uniform(0,1,self.n_links)))	# adding random nosie so that pipes of the same diamter get different ranks in different simulations

		# adding up the repair times to know at what day after the EQ a pipe will be resotred
		temp = np.cumsum(rep_time_pipes_i[rep_rank_pipes])
		temp = temp + (8/24)*np.floor(temp/(16/24)) # adding 8 hours for every 16 hours of repair time bacuse we have 16 hour shifts

		temp_seq = np.argsort(self.link_ID[rep_rank_pipes]) # finding the sequence in which the repair times need to be put back
		rep_time_pipes = temp[temp_seq]* (rep_time_pipes_i!=0) # final repair times for each link
		self.pipe_rep_cost[self.fast,self.r,self.i] = np.sum(self.pipes_leaks)*0.1 + 0.75*np.sum(self.pipes_breaks)
		self.rep_time_pipes = np.ceil(rep_time_pipes)*self.closed_pipes
		self.unique_rep_times_pipes = np.unique(self.rep_time_pipes[self.rep_time_pipes > 0]).astype(int)

	def complete_loop(self):
		self.frac_tax_lots_conn_wpump[self.fast, self.r, self.i, self.unique_rep_times_pipes[self.t]:] = 1
		self.frac_tax_lots_conn_wpump1[self.fast, self.r, self.i, self.unique_rep_times_pipes[self.t]:] = 1
		self.frac_tax_lots_conn_wpump2[self.fast, self.r, self.i, self.unique_rep_times_pipes[self.t]:] = 1
		self.frac_tax_lots_conn_wpump3[self.fast, self.r, self.i, self.unique_rep_times_pipes[self.t]:] = 1

		self.frac_closed_wpipes[self.fast, self.r, self.i, self.unique_rep_times_pipes[self.t]:] = 0
		self.frac_closed_wpipes1[self.fast, self.r, self.i, self.unique_rep_times_pipes[self.t]:] = 0
		self.frac_closed_wpipes2[self.fast, self.r, self.i, self.unique_rep_times_pipes[self.t]:] = 0
		self.frac_closed_wpipes3[self.fast, self.r, self.i, self.unique_rep_times_pipes[self.t]:] = 0
	
	def time_to_p_conn_eval(self):
		if self.frac_tax_lots_conn_wpump[self.fast, self.r, self.i][-1] < 0.9:
			self.time_to_90p_conn_wpump[self.fast, self.r, self.i] = 2000		# 2000 is just any number greater than 365; to show that it takes longer than 365 days
			self.time_to_100p_conn_wpump[self.fast, self.r, self.i] = 2000
		elif self.frac_tax_lots_conn_wpump[self.fast, self.r, self.i][-1] < 1.0:
			self.time_to_100p_conn_wpump[self.fast, self.r, self.i] = 2000

		if self.frac_tax_lots_conn_wpump1[self.fast, self.r, self.i][-1] < 0.9:
			self.time_to_90p_conn_wpump1[self.fast, self.r, self.i] = 2000		# 2000 is just any number greater than 365; to show that it takes longer than 365 days
			self.time_to_100p_conn_wpump1[self.fast, self.r, self.i] = 2000
		elif self.frac_tax_lots_conn_wpump1[self.fast, self.r, self.i][-1] < 1.0:
			self.time_to_100p_conn_wpump1[self.fast, self.r, self.i] = 2000

		if self.frac_tax_lots_conn_wpump2[self.fast, self.r, self.i][-1] < 0.9:
			self.time_to_90p_conn_wpump2[self.fast, self.r, self.i] = 2000		# 2000 is just any number greater than 365; to show that it takes longer than 365 days
			self.time_to_100p_conn_wpump2[self.fast, self.r, self.i] = 2000
		elif self.frac_tax_lots_conn_wpump2[self.fast, self.r, self.i][-1] < 1.0:
			self.time_to_100p_conn_wpump2[self.fast, self.r, self.i] = 2000

		if self.frac_tax_lots_conn_wpump3[self.fast, self.r, self.i][-1] < 0.9:
			self.time_to_90p_conn_wpump3[self.fast, self.r, self.i] = 2000		# 2000 is just any number greater than 365; to show that it takes longer than 365 days
			self.time_to_100p_conn_wpump3[self.fast, self.r, self.i] = 2000
		elif self.frac_tax_lots_conn_wpump3[self.fast, self.r, self.i][-1] < 1.0:
			self.time_to_100p_conn_wpump3[self.fast, self.r, self.i] = 2000

	def plot_results(self):
		plt.figure()
		for rt in range(len(self.RT)):
			temp_avg = np.mean(self.frac_tax_lots_conn_wpump[self.fast, rt], axis=0)
			plt.plot(temp_avg, linewidth = 0.75, label= str(rt))
		plt.legend()
		plt.grid()

	def define_wtp_damage_ratio(self):
		self.damage_ratio_wtp = np.array([0, 0.02, 0.1, 0.5, 1.0])

	def define_wtp_rep_variables(self):
		if self.fast == 0:
			self.wtp_rep_time_mu = np.array([1, 6, 20, 90]) # mean repair time for water treatement plants for DS2-DS5
			self.wtp_rep_time_std = np.array([1, 6, 20, 90]) # std for repair time		
		else:
			self.wtp_rep_time_mu = np.array([1, 6, 20, 90])*0.5 # mean repair time for water treatement plants for DS2-DS5
			self.wtp_rep_time_std = np.array([1, 6, 20, 90])*0.5 # std for repair time	

		self.wtp_rep_time_cov = self.wtp_rep_time_std/self.wtp_rep_time_mu # COV of repiar time
		self.wtp_rep_time_log_med = np.log(self.wtp_rep_time_mu/np.sqrt(self.wtp_rep_time_cov**2+1)) # lognormal parameters for repair time model
		self.wtp_rep_time_beta = np.sqrt(np.log(self.wtp_rep_time_cov**2+1))
		self.wtp_rep_time_covm = self.wtp_rep_time_beta[:,None]*self.wtp_rep_time_beta


	def define_wtp_frag_variables(self):
		n_RTP = len(self.RT)
		# note: the wtp is outside the inundation zone. 
		if self.retrofit == 0:
			self.IM_wtp_d = 0.0*self.wtp_IM[:,self.r + n_RTP]		# first column is link ID; flow depth - units: m
		else:
			self.IM_wtp_d = 0.0*self.wtp_IM[:,self.r + n_RTP] - (5.0/3.2808) # first column is link ID; flow depth - units: m

		self.IM_wtp_d[self.IM_wtp_d<0] = 0
		IM_wtp_v = self.wtp_IM[:,self.r] # first column is link ID; flow speed - units: m/s
		IM_wtp_v[IM_wtp_v<0] = 0

		flow_speed_type = 2 					# moderate flow speed
		if IM_wtp_v <= 1: flow_speed_type = 1	# low flow speed
		if IM_wtp_v >= 5: flow_speed_type = 3	# hight flow speed

		self.wtp_frag_med = np.array([1.8, 4.0, 6.3, 10.0])/(3.2808*self.frag_mod_fac2[0, flow_speed_type-1])
		self.wtp_frag_beta = np.array([0.4, 0.4, 0.4, 0.4]) + 0.1*(IM_wtp_v>1.0)

		self.wtp_frag_covm = self.wtp_frag_beta[:,None]*self.wtp_frag_beta

	def wtp_eval(self):
		Capacity_wtp, _ = DCH.capacity_reptime_gen(self.wtp_frag_med, self.wtp_frag_covm, self.wtp_rep_time_log_med, self.wtp_rep_time_covm, self.n_sims)	# generating correlated capacity estimates for wtp
		_, self.DS_wtp[self.fast, self.retrofit, :, self.r] = DCH.DS_eval_1d(self.IM_wtp_d, Capacity_wtp, 999)
		
		self.rep_time_wtp[self.fast, self.retrofit, :, self.r] = np.array([self.rep_time_wtp_gen[self.fast, self.retrofit, int(i), int(obj)-1] for i, obj in enumerate(self.DS_wtp[self.fast, self.retrofit, :, self.r])])
		self.rep_cost_wtp[self.fast, self.retrofit, :, self.r] = self.damage_ratio_wtp[self.DS_wtp[self.fast, self.retrofit, :, self.r].astype(int)-1]*30000

	def define_wps_damage_ratio(self):
		self.damage_ratio_wps = np.array([0, 0.02, 0.1, 0.5, 1.0])

	def define_wps_rep_variables(self):
		if self.fast == 0:
			self.wps_rep_time_mu = np.array([1, 6, 20, 240]) # mean repair time for water treatement plants for DS2-DS5
			self.wps_rep_time_std = np.array([1, 6, 20, 120]) # std for repair time		
		else:
			self.wps_rep_time_mu = np.array([1, 6, 20, 240])*0.5 # mean repair time for water treatement plants for DS2-DS5
			self.wps_rep_time_std = np.array([1, 6, 20, 120])*0.5 # std for repair time

		self.wps_rep_time_cov = self.wps_rep_time_std/self.wps_rep_time_mu # COV of repiar time
		self.wps_rep_time_log_med = np.log(self.wps_rep_time_mu/np.sqrt(self.wps_rep_time_cov**2+1)) # lognormal parameters for repair time model
		self.wps_rep_time_beta = np.sqrt(np.log(self.wps_rep_time_cov**2+1))
		self.wps_rep_time_covm = self.wps_rep_time_beta[:,None]*self.wps_rep_time_beta


	def define_wps_frag_variables(self):
		n_RTP = len(self.RT)
		if self.retrofit == 0:
			self.IM_wps_d = np.array(self.ps_IM[self.np, self.r + n_RTP])		# first column is link ID; flow depth - units: m
		else:
			self.IM_wps_d = np.array(self.ps_IM[self.np, self.r + n_RTP] - (5.0/3.2808)) # first column is link ID; flow depth - units: m
		
		self.IM_wps_d[self.IM_wps_d<0] = 0
		IM_wps_v = np.array(self.ps_IM[self.np ,self.r]) # first column is link ID; flow speed - units: m/s
		IM_wps_v[IM_wps_v<0] = 0

		flow_speed_type = 2 					# moderate flow speed
		if IM_wps_v <= 1: flow_speed_type = 1	# low flow speed
		if IM_wps_v >= 5: flow_speed_type = 3	# hight flow speed

		self.wps_frag_med = np.array([1.9, 3.7, 7.7, 9.6])/(3.2808*self.frag_mod_fac2[0, flow_speed_type-1])
		self.wps_frag_beta = np.array([0.4, 0.4, 0.4, 0.4]) + 0.1*(IM_wps_v>1.0)
		self.wps_frag_covm = self.wps_frag_beta[:,None]*self.wps_frag_beta

	def wps_eval(self):
		Capacity_wps, _ = DCH.capacity_reptime_gen(self.wps_frag_med, self.wps_frag_covm, self.wps_rep_time_log_med, self.wps_rep_time_covm, self.n_sims)	# generating correlated capacity estimates for wps
		_, self.DS_wps[self.np, self.fast, self.retrofit, :, self.r] = DCH.DS_eval_1d(self.IM_wps_d, Capacity_wps, 999)

		self.rep_time_wps[self.np, self.fast, self.retrofit, :, self.r] = np.array([self.rep_time_wps_gen[self.fast, self.retrofit, int(i), int(obj)-1] for i, obj in enumerate(self.DS_wps[self.np, self.fast, self.retrofit, :, self.r])])
		self.rep_cost_wps[self.np, self.fast, self.retrofit, :, self.r] = self.damage_ratio_wps[self.DS_wps[self.np, self.fast, self.retrofit, :, self.r].astype(int)-1]*150


	def reorg_gis_data(self):
		"""
		note: drs, added this
		"""
		gis_data = np.zeros((len(self.tax_lot_type), len(self.RT) + 1))
		gis_data[:,0] = np.arange(0, len(self.tax_lot_type), 1)
		header = np.empty(len(self.RT) + 1, dtype = object)
		for r in range(len(self.RT)):
			data_temp = self.frac_tax_lots_conn_write[r]
			avg = np.average(data_temp, axis = 1)
			gis_data[:,r+1] = avg
			header[r+1] = 'tsu_' + str(self.RT[r])

		header[0] = 'ID'
		header = list(header)
		header = ', '.join(header)

		csvwrt(gis_data, header, 'water_conn_tsu')


	# ~~~ tertiary and beyond methods ~~~
	def misc_data_conversions(self):
		self.tax_lot_link_ID = np.array(self.tax_lot_info_raw[:,38]).astype(int).flatten() - 1	# roadway link closest to the tax lot
		self.tax_lot_type = np.array(self.tax_lot_info_raw[:,4])									# building type; 0 means no building
		self.logical_tax_lot_type = self.tax_lot_type > 0
		self.tax_lot_pump_region = np.array(self.tax_lot_info_raw[:,40]) - 1 			# information on the pump (region) that serves the tax lot
		self.brittle = [self.ductile == 0]




wdt = water_damage_tsu()
if wdt.__dict__['write_tf'] == True:
	h5wrt(wdt)

plt.show()




