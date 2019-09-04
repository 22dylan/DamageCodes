"""
% this code performs analysis of the road network in Seaside subjected to
% earthquakes
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


# import line_profiler
# lp = line_profiler.LineProfiler()
# from profilestats import profile

# lp_wrapper = lp(do_stuff)
# lp_wrapper(numbers)
# lp.print_stats()

class transportationdamage_earthquake():
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
		self.G = DCH.create_graph(self.adj_road)

	def run(self):
		self.print_runinfo()
		for self.fast in range(self.fast_vals):
			print('fast: {}' .format(self.fast))
			for self.retrofit in range(self.retrofit_vals):
				print('retrofit: {}' .format(self.retrofit))
				# all fragility estimates based on HAZUS
				self.define_roadway_frag_variables()
				self.define_bridge_frag_variables()
				self.define_roadway_rep_variables()
				self.define_bridge_rep_variables()
				for self.r in range(len(self.RT)): 		#loop through return intervals
					self.define_IMs()
					
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
		self.link_IMs = self.readmat('link_IM.mat', 'link_IMs', dtype = 'array')
		self.link_len = self.readmat('link_info.mat', 'link_len', dtype = 'array_flat')
		self.link_unit_cost = self.readmat('link_info', 'link_unit_cost', dtype = 'array_flat')

		self.bridge_fragility_IM_link = self.readmat('bridge_fragility_IM_link.mat', 'bridge_frag_IM', dtype = 'array')

		self.start_node = self.readmat('roads_adjacency.mat', 'start_node', dtype = 'array_flat', idx_0 = True)
		self.end_node = self.readmat('roads_adjacency.mat', 'end_node', dtype = 'array_flat', idx_0 = True)
		self.adj_road = self.readmat('roads_adjacency.mat', 'adj_road', dtype = 'adj_list')

		self.tax_lot_info_raw = self.readmat('tax_lot_info.mat', 'tax_lot_info', dtype = 'array')

		self.misc_data_conversions()

	def initialize_variables(self):
		self.RT = np.array([100, 200, 250, 500 ,1000, 2500, 5000, 10000]) # return period of earthquake events (in years)
		self.hospital_node = np.array([299])						# node number of hospital(s) in the road network
		self.fire_station_node = np.array([228, 229])			# node number of fire station(s) in the road network
		self.time_inst = np.array(range(3*365))+1		# time instances (or, in this case, # days in 3 years)

		self.bridge_cost = [20000, 20000, 1000, 20000, 20000, 20000, 20000, 20000, 20000, 1000, 20000, 20000, 20000] # in thousands of dollars

		self.n_nodes = np.shape(self.adj_road)[0]
		self.n_links = len(self.link_len)
		self.n_bridges = len(self.frag_est)					# number of bridges
		self.time_inst = np.array(range(3*365))+1		# time instances (or, in this case, # days in 3 years)
		self.damage_ratio_bridges = np.array([0.0, 0.03, 0.08, 0.25, 2/3])	# for complete damage the raatio is calculated assuming 3 spans per bridge
		self.damage_ratio_roads = np.array([0.0, 0.05, 0.20, 0.7])			# for complete damage the raatio is calculated assuming 3 spans per bridge
		self.outfilename = 'transportation_damage_eq'


	def preallocate_space(self):
		RT_size = len(self.RT)
		self.time_size = len(self.time_inst) + 1
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

	def define_roadway_frag_variables(self):
		self.roadway_frag_med = np.array([6, 12, 24]) 			# median fragility values - pgd in inches
		self.roadway_frag_beta = np.array([0.7, 0.7, 0.7]) 		# dispersion around fragility values
		self.roadway_frag_covm = self.roadway_frag_beta[:,None]*self.roadway_frag_beta	# roadway fragility covariance matrix		

	def define_bridge_frag_variables(self):
		self.bridge_frag_med = self.frag_est[:]
		self.bridge_frag_beta = np.array([0.6, 0.6, 0.6, 0.6])
		if self.retrofit == 1:
			self.bridge_frag_med = self.bridge_frag_med*1.25		# 25% increase in median fragility estimates
			self.bridge_frag_beta = self.bridge_frag_beta*0.75
		self.bridge_frag_covm = np.dstack([self.bridge_frag_beta[:,None]*self.bridge_frag_beta]*self.n_bridges)	# "stacking" covm matrices

	def define_roadway_rep_variables(self):
		# repair time parameters for roadway links
		self.roadway_rep_time_mu = np.array([0.9, 2.2, 21])					# mean repair time for roads for DS2-DS4
		self.roadway_rep_time_std = np.array([0.05, 1.8, 16])				# std for repair time
		self.roadway_rep_time_cov = self.roadway_rep_time_std/self.roadway_rep_time_mu	# COV of repair time
		self.roadway_rep_time_log_med = np.log(self.roadway_rep_time_mu/np.sqrt(self.roadway_rep_time_cov**2+1)) # lognormal parameters for repair time model
		self.roadway_rep_time_beta = np.sqrt(np.log(self.roadway_rep_time_cov**2+1))
		self.roadway_rep_time_covm = self.roadway_rep_time_beta[:, None]*self.roadway_rep_time_beta

	def define_bridge_rep_variables(self):
		# repair time parameters for bridges
		self.bridge_rep_time_mu = np.array([0.6, 2.5, 75, 230]) 	# mean repair time for roads for DS2-DS5
		self.bridge_rep_time_std = np.array([0.6, 2.7, 42, 110]) # std for repair time
		if self.fast == 1:
			self.bridge_rep_time_mu = self.bridge_rep_time_mu*0.5
			self.bridge_rep_time_std = self.bridge_rep_time_std*0.5
		self.bridge_rep_time_cov = self.bridge_rep_time_std/self.bridge_rep_time_mu # COV of repair time
		self.bridge_rep_time_log_med = np.log(self.bridge_rep_time_mu/np.sqrt(self.bridge_rep_time_cov**2+1)) # lognormal parameters for repair time model
		self.bridge_rep_time_beta = np.sqrt(np.log(self.bridge_rep_time_cov**2+1))
		self.bridge_rep_time_covm = self.bridge_rep_time_beta[:,None]*self.bridge_rep_time_beta

	def define_IMs(self):
		self.IM_road_sa5s = self.link_IMs[:,self.r]	# link ID - units: g
		# IM_roads_PGD = g*IM_road_sa5s*(25/(4*pi*pi))/2.3; % converting Sa at 5s to PGD 	# commented out in original matlab code. still in matlab syntax.
		self.IM_roads_PGD = np.zeros(np.size(self.IM_road_sa5s))			# PGD (permanent ground deformation) from Shafiq's analysis is zero
		self.IM_bridge_sa1s = self.bridge_frag_IM[:,self.r]					# fragilities

	def conn_analysis1(self):
		np.random.seed(self.i)
		Capacity_roads, rep_time_roads_all = DCH.capacity_reptime_gen(self.roadway_frag_med, self.roadway_frag_covm, self.roadway_rep_time_log_med, self.roadway_rep_time_covm, self.n_links)
		np.random.seed(self.i)
		Capacity_bridges, rep_time_bridges_all = DCH.capacity_reptime_gen_2d(self.bridge_frag_med, self.bridge_frag_covm, self.bridge_rep_time_log_med, self.bridge_rep_time_covm, self.n_bridges)

		self.closed_roads, DS_roads = DCH.DS_eval(self.IM_roads_PGD, Capacity_roads, 3)	#damage state and closure of roads
		self.closed_roads[self.bridge_link_ID] = False	# these links are bridges so their closure decision is based on bridge damage state
		
		closed_bridges, DS_bridges = DCH.DS_eval(self.IM_bridge_sa1s, Capacity_bridges, 2)	#damage state and closure of bridges
		self.closed_roads[self.bridge_link_ID[closed_bridges]] = True								# closed bridges within the closed roads array
		
		self.frac_closed_roads[self.fast, self.retrofit, self.r, self.i, :] = np.mean(self.closed_roads)*np.ones(self.time_size)
		self.closed_roads_ID = np.where(self.closed_roads)[0]

		self.rep_time_roads = np.array([rep_time_roads_all[int(i), int(obj)-1] for i, obj in enumerate(DS_roads)])
		rep_time_bridges = np.array([rep_time_bridges_all[int(i), int(obj)-1] for i, obj in enumerate(DS_bridges)])

		# all costs in 1000s of dollars
		self.rep_cost_bridges[self.fast, self.retrofit, self.r, self.i] = np.sum(self.damage_ratio_bridges[DS_bridges.astype(int)-1] * self.bridge_cost)
		self.rep_cost_roads[self.fast, self.retrofit, self.r, self.i] = np.sum(self.damage_ratio_roads[DS_roads.astype(int)-1]*self.link_len*self.link_unit_cost*0.0003048)
		self.tot_cost_road_bridges[self.fast, self.retrofit, self.r, self.i] = self.rep_cost_bridges[self.fast, self.retrofit, self.r, self.i] + self.rep_cost_roads[self.fast, self.retrofit, self.r, self.i]

		self.rep_time_roads[self.bridge_link_ID] = rep_time_bridges
		self.rep_time_roads = np.ceil(self.rep_time_roads) * self.closed_roads

		# creating post-earthquake adjacency matrix - at day 0
		G_post = self.G.copy()
		if len(self.closed_roads_ID) > 0:
			G_post = DCH.delete_edges(G_post, self.start_node, self.end_node, self.closed_roads_ID)
		self.bins = DCH.conn_comp(G_post, self.n_nodes)
		self.link_failure[self.r, self.closed_roads_ID, self.i] = 1

	def conn_analysis2(self):
		self.closed_roads[(self.rep_time_roads <= self.unique_reptime_roads[self.t])] = False
		self.closed_roads_ID = np.where(self.closed_roads)[0]
		self.frac_closed_roads[self.fast, self.retrofit, self.r, self.i, self.unique_reptime_roads[self.t]] = np.mean(self.closed_roads)

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
		self.frag_est = np.array(self.bridge_fragility_IM_link[:,0:4])							# fragility estimates (Sa at 1s)
		self.bridge_frag_IM = np.array(self.bridge_fragility_IM_link[:,4:12])						# IMs for 8 return period events
		self.bridge_link_ID = np.array(self.bridge_fragility_IM_link[:,-1]).astype(int) - 1		# link ID of roadway link on which bridge is situated (0-indexed)
		
		self.tax_lot_link_ID = np.array(self.tax_lot_info_raw[:,37]).astype(int).flatten() - 1	# roadway link closest to the tax lot
		self.tax_lot_type = np.array(self.tax_lot_info_raw[:,4])									# building type; 0 means no building
		self.tax_lot_start_node = self.start_node[self.tax_lot_link_ID]
		self.tax_lot_end_node = self.end_node[self.tax_lot_link_ID]


tde = transportationdamage_earthquake()


if tde.__dict__['write_tf'] == True:
	h5wrt(tde)

plt.show()












