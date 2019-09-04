"""
% this code performs analysis of the road network in Seaside subjected to
% tsunamis and earthquakes
% The main objective is to determine the connectivty of the tax lot
% locations to the hospital and the fire stattion

original code written by Sabarethinam Kameshwar in Matlab
converted from Matlab to python by Dylan R. Sanderson
Nov. 2018

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


class transportationdamage_tsu_eq():
	def __init__(self):	
		self.setup()		# reading input files, initializing variables, etc. 
		self.run()	

	# ~~~ user input up top ~~~
	def user_input_vals(self):
		self.n_sims = 10000			# number of simulations
		self.fast_vals = 1			# originally set to 2
		self.retrofit_vals = 1		# originally set to 4
		
		self.plot_tf = False
		self.write_tf = False

	# ~~~ main methods ~~~
	def setup(self):
		self.user_input_vals()
		self.load_mat_files()
		self.initialize_variables()
		self.preallocate_space()
		self.G = DCH.create_graph(self.adj_road)

	def run(self):
		self.print_runinfo()
		for self.fast in range(self.fast_vals):			# loop through fast repair time values
			print('fast: {}' .format(self.fast))
			for self.retrofit in range(self.retrofit_vals):	# loop through retrofit values
				print('retrofit: {}' .format(self.retrofit))
				self.define_bridge_frag_variables(tsu_eq = 'earthquake')
				self.define_roadway_frag_variables(tsu_eq = 'earthquake')
				self.define_roadway_rep_variables()
				self.define_bridge_rep_variables()

				for self.r in range(len(self.RT)): 		#loop through return intervals
					self.define_IMs()
					self.define_flowspeeds(road_bridge = 'road')
					self.define_roadway_frag_variables(tsu_eq = 'tsunami')
					self.define_flowspeeds(road_bridge = 'bridge')
					self.define_bridge_frag_variables(tsu_eq = 'tsunami')

					timer = time.time()
					for self.i in range(self.n_sims):
						self.conn_analysis1()
						self.fire_hosp_conn(init_loop=True)

						self.unique_reptime_roads = np.unique(self.rep_time_roads[(self.rep_time_roads>0) & (self.rep_time_roads<=max(self.time_inst))]).astype(int)
						for self.t in range(len(self.unique_reptime_roads)):
							self.conn_analysis2()
							self.fire_hosp_conn(init_loop = False)
							
							if  len(self.closed_roads_ID) == 0:
								self.complete_loop()
								break
						
						self.time_to_p_conn_eval()

					print('elapsed time: {}' .format(time.time() - timer))
				if self.plot_tf == True:
					self.plot_results()


	# ~~~ secondary methods ~~~
	#  ~~ from self.setup() ~~
	def load_mat_files(self):
		self.link_IM_d = self.readmat('link_IM_tsu.mat', 'link_IM_d', dtype = 'array')
		self.link_IM_v = self.readmat('link_IM_tsu.mat', 'link_IM_v', dtype = 'array')
		
		self.link_len = self.readmat('link_info.mat', 'link_len', dtype = 'array_flat')
		self.link_unit_cost = self.readmat('link_info.mat', 'link_unit_cost', dtype = 'array_flat')

		self.bridge_IM_v = self.readmat('bridge_fragility_IM_link_tsu.mat', 'bridge_IM_v', dtype = 'array')
		self.bridge_frag_IM_d = self.readmat('bridge_fragility_IM_link_tsu.mat', 'bridge_frag_IM_d' , dtype = 'array')

		self.start_node = self.readmat('roads_adjacency.mat', 'start_node', dtype = 'array_flat', idx_0 = True)
		self.end_node = self.readmat('roads_adjacency.mat', 'end_node', dtype = 'array_flat', idx_0 = True)
		self.adj_road = self.readmat('roads_adjacency.mat', 'adj_road', dtype = 'adj_list')

		self.tax_lot_info_raw = self.readmat('tax_lot_info.mat', 'tax_lot_info', dtype = 'none')

		self.link_IMs = self.readmat('link_IM_eq.mat', 'link_IMs', dtype = 'array')
		self.bridge_frag_IM = self.readmat('bridge_fragility_IM_link_eq.mat', 'bridge_frag_IM', dtype='array')
		# self.bridge_fragility_IM_link = self.readmat('bridge_fragility_IM_link_eq.mat', 'bridge_frag_IM', dtype = 'array')

		self.misc_data_conversions()

	def initialize_variables(self):
		self.RT = np.array([100, 200, 250, 500 ,1000, 2500, 5000, 10000]) # return period of tsunami events (in years)
		self.hospital_node = np.array([299])						# node number of hospital(s) in the road network
		self.fire_station_node = np.array([228, 229])			# node number of fire station(s) in the road network
		
		self.bridge_cost = [20000, 20000, 1000, 20000, 20000, 20000, 20000, 20000, 20000, 1000, 20000, 20000, 20000] # in thousands of dollars
		
		# factors that modify the meadian fragilites to account for flow rates and debris; first row is without large debris and the second one is with considering large debris
		#	the factors above are for components that with high vulnerability to flows and moderate vulnerability to debris - bridges
		self.frag_mod_fac_b = np.array([[1.0, 1.5, 2.0], [1.2, 1.8, 2.4]])
		# these factors are for components that are moderaltely vulnerable to flow and less to desbris -- roads
		self.frag_mod_fac_r = np.array([[1.0, 1.4, 1.6], [1.1, 1.5, 1.8]])

		self.n_nodes = np.shape(self.adj_road)[0]
		self.n_links = len(self.link_len)
		self.n_bridges = len(self.bridge_frag_IM_d[:,0])					# number of bridges
		self.time_inst = np.array(range(3*365))+1		# time instances (or, in this case, # days in 3 years)
		
		self.damage_ratio_bridges_tsu = np.array([0.0, 0.02, 0.10, 0.50, 1.00])	# for complete damage the raatio is calculated assuming 3 spans per bridge
		self.damage_ratio_roads_tsu = np.array([0.0, 0.02, 0.10, 0.50, 1.00])			# for complete damage the raatio is calculated assuming 3 spans per bridge
		self.damage_ratio_bridges_eq = np.array([0.0, 0.03, 0.08, 0.25, 2/3])
		self.damage_ratio_roads_eq = np.array([0.0, 0.05, 0.20, 0.7])
		self.outfilename = 'transportation_damage_tsu_eq'

	def preallocate_space(self):
		RT_size = len(self.RT)
		self.time_size = len(self.time_inst) + 1

		self.rep_cost_bridges_tsu = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)
		self.rep_cost_roads_tsu = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)
		self.tot_cost_road_bridges_tsu = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)

		self.rep_cost_bridges_eq = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)
		self.rep_cost_roads_eq = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)
		self.tot_cost_road_bridges_eq = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)

		self.rep_cost_bridges = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)
		self.rep_cost_roads = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)
		self.tot_cost_road_bridges = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)

		self.frac_closed_roads = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims*self.time_size).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims, self.time_size)
		self.frac_tax_lots_conn_fire_hosp = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims*self.time_size).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims, self.time_size)

		self.time_to_90p_conn_fire_hosp = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)
		self.time_to_100p_conn_fire_hosp = np.zeros(self.fast_vals*self.retrofit_vals*RT_size*self.n_sims).reshape(self.fast_vals, self.retrofit_vals, RT_size, self.n_sims)

		self.link_failure = np.zeros((RT_size, self.n_links, self.n_sims))


	#  ~~ from self.run() ~~ 
	def print_runinfo(self):
		print('n_sims: {}' .format(self.n_sims))
		print('fast_vals: {}' .format(self.fast_vals))
		print('retrofit_vals: {}' .format(self.retrofit_vals))
		print('plot results: {}' .format(self.plot_tf))
		print('write results: {}\n' .format(self.write_tf))

	def define_bridge_frag_variables(self, tsu_eq):
		if tsu_eq == 'earthquake':
			self.bridge_frag_med_eq = self.bridge_frag_IM[:,0:4]
			self.bridge_frag_beta_eq = np.array([0.6, 0.6, 0.6, 0.6])

			if (self.retrofit == 1) or (self.retrofit == 3):
				self.bridge_frag_med_eq = self.bridge_frag_med_eq*1.25		# 25% increase in median fragility estimates
				self.bridge_frag_beta_eq = self.bridge_frag_beta_eq*0.75
			self.bridge_frag_covm_eq = np.dstack([self.bridge_frag_beta_eq[:,None]*self.bridge_frag_beta_eq]*self.n_bridges)	# "stacking" covm matrices
		
		elif tsu_eq == 'tsunami':
			self.bridge_frag_med_tsu = np.zeros((self.n_bridges,len(self.bridge_rep_time_mu_tsu)))
			self.bridge_frag_covm_tsu = np.zeros(len(self.bridge_rep_time_mu_tsu)*len(self.bridge_rep_time_mu_tsu)*self.n_bridges).reshape(len(self.bridge_rep_time_mu_tsu), len(self.bridge_rep_time_mu_tsu), self.n_bridges)

			for nb in range(self.n_bridges):
				if self.flow_speed_type[nb] == 1:
					self.bridge_frag_med_tsu[nb,:] = self.bridge_frag_IM_d[nb,0:4]/(3.2808*self.frag_mod_fac_b[0,0])
					self.bridge_frag_beta_tsu = np.array([0.4, 0.4, 0.4, 0.4])
				elif self.flow_speed_type[nb]==2:
					self.bridge_frag_med_tsu[nb,:] = self.bridge_frag_IM_d[nb,0:4]/(3.2808*self.frag_mod_fac_b[0,1])
					self.bridge_frag_beta_tsu = np.array([0.5, 0.5, 0.5, 0.5])
				elif self.flow_speed_type[nb]==3:
					self.bridge_frag_med_tsu[nb,:] = self.bridge_frag_IM_d[nb,0:4]/(3.2808*self.frag_mod_fac_b[0,2])
					self.bridge_frag_beta_tsu = np.array([0.5, 0.5, 0.5, 0.5])
				self.bridge_frag_covm_tsu[:,:,nb] = self.bridge_frag_beta_tsu*self.bridge_frag_beta_tsu

			if (self.retrofit == 2) or (self.retrofit==3):
				self.bridge_frag_med_tsu = self.bridge_frag_med_tsu*1.25 	#25% increase in median fragility estimates
				self.bridge_frag_covm_tsu = self.bridge_frag_covm_tsu*(0.75*0.75)		

	def define_roadway_frag_variables(self, tsu_eq):
		if tsu_eq == 'earthquake':
			self.roadway_frag_med_eq = np.array([6, 12, 24])
			self.roadway_frag_beta_eq = np.array([0.7, 0.7, 0.7])
			self.roadway_frag_covm_eq= self.roadway_frag_beta_eq[:,None]*self.roadway_frag_beta_eq	# roadway fragility covariance matrix		

		elif tsu_eq == 'tsunami':
			self.roadway_frag_med_uf = np.array([2.2, 4.2, 6.8, 11])/(39.37/12) # median fragility values - tsunami height in feet
			self.roadway_frag_med_tsu = np.zeros(len(self.flow_speed_type)*len(self.roadway_frag_med_uf)).reshape(len(self.flow_speed_type), len(self.roadway_frag_med_uf))
			self.roadway_frag_med_tsu[self.flow_speed_type==1,:] = np.tile(self.roadway_frag_med_uf/self.frag_mod_fac_r[0,0], (np.sum(self.flow_speed_type==1), 1))
			self.roadway_frag_med_tsu[self.flow_speed_type==2,:] = np.tile(self.roadway_frag_med_uf/self.frag_mod_fac_r[0,1], (np.sum(self.flow_speed_type==2), 1))
			self.roadway_frag_med_tsu[self.flow_speed_type==3,:] = np.tile(self.roadway_frag_med_uf/self.frag_mod_fac_r[0,2], (np.sum(self.flow_speed_type==3), 1))

			self.roadway_frag_beta_tsu	 = np.zeros(np.shape(self.roadway_frag_med_tsu))
			self.roadway_frag_beta_tsu[self.flow_speed_type==1,:] = np.tile([0.4, 0.4, 0.4, 0.4], (np.sum(self.flow_speed_type==1), 1))
			self.roadway_frag_beta_tsu[self.flow_speed_type>1,:] = np.tile([0.5, 0.5, 0.5, 0.5], (np.sum(self.flow_speed_type>1), 1))

			self.roadway_frag_covm_tsu = np.zeros(len(self.roadway_frag_med_uf)*len(self.roadway_frag_med_uf)*self.n_links).reshape(len(self.roadway_frag_med_uf), len(self.roadway_frag_med_uf), self.n_links)
			for nl in range(self.n_links):
				self.roadway_frag_covm_tsu[:,:,nl] = self.roadway_frag_beta_tsu[nl,:]*self.roadway_frag_beta_tsu[nl,:]

	def define_roadway_rep_variables(self):
		# tsunami repair times
		# repair time parameters for roadway links
		self.roadway_rep_time_mu_tsu = np.array([1, 3, 20, 30])					# mean repair time for roads for DS2-DS4
		self.roadway_rep_time_std_tsu = np.array([0.5, 0.5, 0.5, 0.5])*self.roadway_rep_time_mu_tsu				# std for repair time
		self.roadway_rep_time_cov_tsu = self.roadway_rep_time_std_tsu/self.roadway_rep_time_mu_tsu	# COV of repair time
		self.roadway_rep_time_log_med_tsu = np.log(self.roadway_rep_time_mu_tsu/np.sqrt(self.roadway_rep_time_cov_tsu**2+1)) # lognormal parameters for repair time model
		self.roadway_rep_time_beta_tsu = np.sqrt(np.log(self.roadway_rep_time_cov_tsu**2+1))
		self.roadway_rep_time_covm_tsu = self.roadway_rep_time_beta_tsu[:, None]*self.roadway_rep_time_beta_tsu
		
		# EQ repair times
		# repair time parameters for roadway links
		self.roadway_rep_time_mu_eq = np.array([0.9, 2.2, 21])					# mean repair time for roads for DS2-DS4
		self.roadway_rep_time_std_eq = np.array([0.05, 1.8, 16])				# std for repair time
		self.roadway_rep_time_cov_eq = self.roadway_rep_time_std_eq/self.roadway_rep_time_mu_eq	# COV of repair time
		self.roadway_rep_time_log_med_eq = np.log(self.roadway_rep_time_mu_eq/np.sqrt(self.roadway_rep_time_cov_eq**2+1)) # lognormal parameters for repair time model
		self.roadway_rep_time_beta_eq = np.sqrt(np.log(self.roadway_rep_time_cov_eq**2+1))
		self.roadway_rep_time_covm_eq = self.roadway_rep_time_beta_eq[:, None]*self.roadway_rep_time_beta_eq

	def define_bridge_rep_variables(self):
		# repair time parameters for bridges
		self.bridge_rep_time_mu_tsu = np.array([1, 4, 30, 120]) 	# mean repair time for roads for DS2-DS5
		self.bridge_rep_time_std_tsu = np.array([0.5, 0.5, 0.5, 0.5])*self.bridge_rep_time_mu_tsu # std for repair time
		if self.fast == 1:
			self.bridge_rep_time_mu_tsu = self.bridge_rep_time_mu_tsu*0.5
			self.bridge_rep_time_std_tsu = self.bridge_rep_time_std_tsu*0.5
		self.bridge_rep_time_cov_tsu = self.bridge_rep_time_std_tsu/self.bridge_rep_time_mu_tsu # COV of repair time
		self.bridge_rep_time_log_med_tsu = np.log(self.bridge_rep_time_mu_tsu/np.sqrt(self.bridge_rep_time_cov_tsu**2+1)) # lognormal parameters for repair time model
		self.bridge_rep_time_beta_tsu = np.sqrt(np.log(self.bridge_rep_time_cov_tsu**2+1))
		self.bridge_rep_time_covm_tsu = self.bridge_rep_time_beta_tsu[:,None]*self.bridge_rep_time_beta_tsu

		# repair time parameters for bridges
		self.bridge_rep_time_mu_eq = np.array([0.6, 2.5, 75, 230]) 	# mean repair time for roads for DS2-DS5
		self.bridge_rep_time_std_eq = np.array([0.6, 2.7, 42, 110]) # std for repair time
		if self.fast == 1:
			self.bridge_rep_time_mu_eq = self.bridge_rep_time_mu_eq*0.5
			self.bridge_rep_time_std_eq = self.bridge_rep_time_std_eq*0.5
		self.bridge_rep_time_cov_eq = self.bridge_rep_time_std_eq/self.bridge_rep_time_mu_eq # COV of repair time
		self.bridge_rep_time_log_med_eq = np.log(self.bridge_rep_time_mu_eq/np.sqrt(self.bridge_rep_time_cov_eq**2+1)) # lognormal parameters for repair time model
		self.bridge_rep_time_beta_eq = np.sqrt(np.log(self.bridge_rep_time_cov_eq**2+1))
		self.bridge_rep_time_covm_eq = self.bridge_rep_time_beta_eq[:,None]*self.bridge_rep_time_beta_eq

	def define_IMs(self):
		self.IM_roads_d = self.link_IM_d[:,self.r+1]			# first column is link ID - units: m
		self.IM_roads_v = self.link_IM_v[:,self.r+1]			# first column is link ID - units: m
		self.IM_bridge_d = self.bridge_frag_IM_d[:,self.r+4]	# first 4 columns are fragilities
		self.IM_bridge_v = self.bridge_IM_v[:,self.r]

		self.IM_road_sa5s = self.link_IMs[:,self.r]	# link ID - units: g
		# IM_roads_PGD = g*IM_road_sa5s*(25/(4*pi*pi))/2.3; % converting Sa at 5s to PGD 	# commented out in original matlab code. still in matlab syntax.
		self.IM_roads_PGD = np.zeros(np.size(self.IM_road_sa5s))			# PGD (permanent ground deformation) from Shafiq's analysis is zero
		self.IM_bridge_sa1s = self.bridge_frag_IM[:,self.r+4]					# fragilities

	def define_flowspeeds(self, road_bridge):
		if road_bridge == 'road':
			self.flow_speed_type = 2*np.ones(self.n_links) 			# moderate flow speed
			self.flow_speed_type[self.IM_roads_v<=1.0] = 1	# low flow speed
			self.flow_speed_type[self.IM_roads_v>=5.0] = 3	# high flow speed
		elif road_bridge == 'bridge':
			self.flow_speed_type = 2*np.ones(self.n_bridges) 	# moderate flow speed
			self.flow_speed_type[self.IM_bridge_v<=1.0] = 1	# low flow speed
			self.flow_speed_type[self.IM_bridge_v>=5.0] = 3	# high flow speed

	def closed_roads_eval(self, DS, DS_item, per_fail, road_bridge):
		np.random.seed(self.i)
		if road_bridge == 'road':
			for i, obj in enumerate(DS):
				self.closed_roads[DS_item == obj] = np.random.uniform(0,1,(np.sum(DS_item == obj))) <= per_fail[i]
		elif road_bridge == 'bridge':
			for i, obj in enumerate(DS):
				self.closed_roads[self.bridge_link_ID[DS_item == obj]] = np.random.uniform(0,1,(np.sum(DS_item == obj))) <= per_fail[i]

	def conn_analysis1(self):	
		# tsunami
		np.random.seed(self.i)
		Capacity_roads_tsu, rep_time_roads_all_tsu = DCH.capacity_reptime_gen_2d(self.roadway_frag_med_tsu, self.roadway_frag_covm_tsu, self.roadway_rep_time_log_med_tsu, self.roadway_rep_time_covm_tsu, self.n_links)
		np.random.seed(self.i)
		Capacity_bridges_tsu, rep_time_bridges_all_tsu = DCH.capacity_reptime_gen_2d(self.bridge_frag_med_tsu, self.bridge_frag_covm_tsu, self.bridge_rep_time_log_med_tsu, self.bridge_rep_time_covm_tsu, self.n_bridges)

		DS_levels = [2,3,4,5]
		per_fail = [0.02, 0.20, 0.60, 1.00]	# assuming that there is 2%, 20%, 60% and 100% chance of road closure for slight, moderate, extensive, and complete damage respectively.

		self.closed_roads, DS_roads_tsu = DCH.DS_eval(self.IM_roads_d, Capacity_roads_tsu, 999)	# set to 999 to return all "false"
		self.closed_roads_eval(DS_levels, DS_roads_tsu, per_fail, 'road')
		self.closed_roads[self.bridge_link_ID] = False	# these links are bridges so their closure decision is based on bridge damage state
		
		_, DS_bridges_tsu = DCH.DS_eval(self.IM_bridge_d, Capacity_bridges_tsu, 999)	#damage state and closure of bridges
		self.closed_roads_eval(DS_levels, DS_bridges_tsu, per_fail, 'bridge')

		rep_time_roads_tsu = np.array([rep_time_roads_all_tsu[int(i), int(obj)-1] for i, obj in enumerate(DS_roads_tsu)])
		rep_time_bridges_tsu = np.array([rep_time_bridges_all_tsu[int(i), int(obj)-1] for i, obj in enumerate(DS_bridges_tsu)])
		rep_time_roads_tsu[self.bridge_link_ID] = rep_time_bridges_tsu

		# all costs in 1000s of dollars
		self.rep_cost_bridges_tsu[self.fast, self.retrofit, self.r, self.i] = np.sum(self.damage_ratio_bridges_tsu[DS_bridges_tsu.astype(int)-1] * self.bridge_cost)
		self.rep_cost_roads_tsu[self.fast, self.retrofit, self.r, self.i] = np.sum(self.damage_ratio_roads_tsu[DS_roads_tsu.astype(int)-1]*self.link_len*self.link_unit_cost*0.0003048)
		self.tot_cost_road_bridges_tsu[self.fast, self.retrofit, self.r, self.i] = self.rep_cost_bridges_tsu[self.fast, self.retrofit, self.r, self.i] + self.rep_cost_roads_tsu[self.fast, self.retrofit, self.r, self.i]

		# EQ
		np.random.seed(self.i)
		Capacity_roads_eq, rep_time_roads_all_eq = DCH.capacity_reptime_gen(self.roadway_frag_med_eq, self.roadway_frag_covm_eq, self.roadway_rep_time_log_med_eq, self.roadway_rep_time_covm_eq, self.n_links)
		np.random.seed(self.i)
		Capacity_bridges_eq, rep_time_bridges_all_eq = DCH.capacity_reptime_gen_2d(self.bridge_frag_med_eq, self.bridge_frag_covm_eq, self.bridge_rep_time_log_med_eq, self.bridge_rep_time_covm_eq, self.n_bridges)
		
		closed_roads_eq, DS_roads_eq = DCH.DS_eval(self.IM_roads_PGD, Capacity_roads_eq, 3)	#damage state and closure of roads
		self.closed_roads[closed_roads_eq] = True

		closed_bridges_eq, DS_bridges_eq = DCH.DS_eval(self.IM_bridge_sa1s, Capacity_bridges_eq, 2)	#damage state and closure of bridges
		self.closed_roads[self.bridge_link_ID[closed_bridges_eq]] = True								# closed bridges within the closed roads array

		rep_time_roads_eq = np.array([rep_time_roads_all_eq[int(i), int(obj)-1] for i, obj in enumerate(DS_roads_eq)])
		rep_time_bridges_eq = np.array([rep_time_bridges_all_eq[int(i), int(obj)-1] for i, obj in enumerate(DS_bridges_eq)])
		rep_time_roads_eq[self.bridge_link_ID] = rep_time_bridges_eq

		self.rep_cost_bridges_eq[self.fast, self.retrofit, self.r, self.i] = np.sum(self.damage_ratio_bridges_eq[DS_bridges_eq.astype(int)-1] * self.bridge_cost)
		self.rep_cost_roads_eq[self.fast, self.retrofit, self.r, self.i] = np.sum(self.damage_ratio_roads_eq[DS_roads_eq.astype(int)-1]*self.link_len*self.link_unit_cost*0.0003048)
		self.tot_cost_road_bridges_eq[self.fast, self.retrofit, self.r, self.i] = self.rep_cost_bridges_eq[self.fast, self.retrofit, self.r, self.i] + self.rep_cost_roads_eq[self.fast, self.retrofit, self.r, self.i]

		# overall analysis
		self.frac_closed_roads[self.fast, self.retrofit, self.r, self.i, :] = np.mean(self.closed_roads)
		self.closed_roads_ID = np.where(self.closed_roads)[0]
		self.rep_time_roads = np.ceil((np.amax((rep_time_roads_tsu, rep_time_roads_eq), axis=0))) * self.closed_roads

		self.rep_cost_bridges[self.fast, self.retrofit, self.r, self.i] = np.sum(np.amax((self.damage_ratio_bridges_eq[DS_bridges_eq.astype(int)-1] * self.bridge_cost, self.damage_ratio_bridges_tsu[DS_bridges_tsu.astype(int)-1] * self.bridge_cost), axis=0))
		self.rep_cost_roads[self.fast, self.retrofit, self.r, self.i] = np.sum(np.amax((self.damage_ratio_roads_eq[DS_roads_eq.astype(int)-1]*self.link_len*self.link_unit_cost*0.0003048, self.damage_ratio_roads_tsu[DS_roads_tsu.astype(int)-1]*self.link_len*self.link_unit_cost*0.0003048), axis=0))
		self.tot_cost_road_bridges[self.fast, self.retrofit, self.r, self.i] = self.rep_cost_bridges[self.fast, self.retrofit, self.r, self.i] + self.rep_cost_roads[self.fast, self.retrofit, self.r, self.i]
		
		# creating post-event adjacency matrix - at day 0
		G_post = self.G.copy()
		if len(self.closed_roads_ID) > 0:
			G_post = DCH.delete_edges(G_post, self.start_node, self.end_node, self.closed_roads_ID)
		self.bins = DCH.conn_comp(G_post, self.n_nodes)
		self.link_failure[self.r, self.closed_roads_ID, self.i] = 1

	def conn_analysis2(self):
		self.closed_roads[(self.rep_time_roads <= self.unique_reptime_roads[self.t])] = False
		self.closed_roads_ID = np.where(self.closed_roads)[0]
		self.frac_closed_roads[self.fast, self.retrofit, self.r, self.i, self.unique_reptime_roads[self.t]:] = np.mean(self.closed_roads)

		# creating post-event adjacency matrix - at day 0
		G_post = self.G.copy()
		if len(self.closed_roads_ID) > 0:
			G_post = DCH.delete_edges(G_post, self.start_node, self.end_node, self.closed_roads_ID)
		self.bins = DCH.conn_comp(G_post, self.n_nodes)

	def fire_hosp_conn(self, init_loop = False):
		fire_hosp_bins = np.concatenate((self.bins[self.fire_station_node], self.bins[self.hospital_node]))	# there can be a maximum of two such bins
		nodes_conn_fire_hosp = np.logical_and((np.logical_or((self.bins==fire_hosp_bins[0]), (self.bins==fire_hosp_bins[1]))), self.bins==fire_hosp_bins[2])		# nodes connected to the fire stattion
		tax_lots_conn_fire_hosp = np.logical_and(nodes_conn_fire_hosp[self.tax_lot_start_node], nodes_conn_fire_hosp[self.tax_lot_end_node])				# tax lots connected to the fire station

		if init_loop == True:
			self.frac_tax_lots_conn_fire_hosp[self.fast, self.retrofit, self.r, self.i] = (np.sum(tax_lots_conn_fire_hosp*(self.tax_lot_type>0))/np.sum((self.tax_lot_type>0)))*np.ones(self.time_size)

		else:
			self.frac_tax_lots_conn_fire_hosp[self.fast, self.retrofit, self.r, self.i, self.unique_reptime_roads[self.t]:] = np.sum(tax_lots_conn_fire_hosp*(self.tax_lot_type>0))/np.sum((self.tax_lot_type>0))

			if (self.frac_tax_lots_conn_fire_hosp[self.fast, self.retrofit, self.r, self.i, self.unique_reptime_roads[self.t]] > 0.9) & (self.frac_tax_lots_conn_fire_hosp[self.fast, self.retrofit, self.r, self.i, self.unique_reptime_roads[self.t]-1] < 0.9):
				self.time_to_90p_conn_fire_hosp[self.fast, self.retrofit, self.r, self.i] = self.unique_reptime_roads[self.t]
			if (self.frac_tax_lots_conn_fire_hosp[self.fast, self.retrofit, self.r, self.i, self.unique_reptime_roads[self.t]] == 1) & (self.frac_tax_lots_conn_fire_hosp[self.fast, self.retrofit, self.r, self.i, self.unique_reptime_roads[self.t]-1] < 1):
				self.time_to_100p_conn_fire_hosp[self.fast, self.retrofit, self.r, self.i] = self.unique_reptime_roads[self.t]

	def complete_loop(self):
		self.frac_tax_lots_conn_fire_hosp[self.fast, self.retrofit, self.r, self.i, self.unique_reptime_roads[self.t]:] = 1
		self.frac_closed_roads[self.fast, self.retrofit, self.r, self.i, self.unique_reptime_roads[self.t]:] = 0

	def time_to_p_conn_eval(self):
		if self.frac_tax_lots_conn_fire_hosp[self.fast, self.retrofit, self.r, self.i][-1] < 0.9:
			self.time_to_90p_conn_fire_hosp[self.fast, self.retrofit, self.r, self.i] = 2000		# 2000 is just any number greater than 365; to show that it takes longer than 365 days
			self.time_to_100p_conn_fire_hosp[self.fast, self.retrofit, self.r, self.i] = 2000
		elif self.frac_tax_lots_conn_fire_hosp[self.fast, self.retrofit, self.r, self.i][-1] < 1.0:
			self.time_to_100p_conn_fire_hosp[self.fast, self.retrofit, self.r, self.i] = 2000

	def plot_results(self):
		plt.figure()
		for rt in range(len(self.RT)):
			temp_avg = np.mean(self.frac_tax_lots_conn_fire_hosp[self.fast, self.retrofit, rt], axis=0)
			plt.plot(temp_avg, linewidth = 0.75, label= str(rt))
		plt.legend()
		plt.grid()

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
		self.bridge_link_ID = np.array(self.bridge_frag_IM_d[:,-1]).astype(int) - 1		# link ID of roadway link on which bridge is situated (0-indexed)
		
		self.tax_lot_link_ID = np.array(self.tax_lot_info_raw[:,37]).astype(int).flatten() - 1	# roadway link closest to the tax lot
		self.tax_lot_type = np.array(self.tax_lot_info_raw[:,4])									# building type; 0 means no building
		self.tax_lot_start_node = self.start_node[self.tax_lot_link_ID]
		self.tax_lot_end_node = self.end_node[self.tax_lot_link_ID]

		# self.bridge_frag_IM = np.array(self.bridge_fragility_IM_link[:,4:12])						# IMs for 8 return period events
		# self.bridge_link_ID = np.array(self.bridge_fragility_IM_link[:,-1]).astype(int) - 1		# link ID of roadway link on which bridge is situated (0-indexed)
		


tdte = transportationdamage_tsu_eq()


if tdte.__dict__['write_tf'] == True:
	h5wrt(tdte)



plt.show()












