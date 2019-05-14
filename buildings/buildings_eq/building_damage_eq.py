"""

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loss Assesment of Seaside, OR
% Mohammad Shafiqual Alam & Andre. R. Barbosa
% Date: 8/25/2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data Description
% col 1= x-utm (Centroid)
% col 2= y-utm (Centroid)
% col 3= Fid NEW (Unique number for each cell)
% col 4= Year information from taxlot
% col 5= Building type (0~6), [No building, W1, W2, C1, C2, C3, S1]
% col 6= Num. of Floors (1~9)
% col 7= Origin of data (1~3),  [1: From Taxlot, 2: Google map, 3: Field survey] 
% col 8= Class_taxlot
% col 9= Year Classification (0, 1, 2, 3),  [Pre, Low, Moderate, High]
    % W2, C1~3, S1~2 (W1)
    % Pre: < 1979
    % Low: 1979 =< < 1995 (1998)
    % Mod: 1995 =< < 2003 (2008)
    % High: >= 2003 (2008)
% col 10=RMV (Real Market Vale)
% col 11=RMV_imporved
% col 12=RMV_land
% col 13= AREA (m^2) (Only from Taxlot, it may overestimate in many cases)
% col 14= HAZUS BUILDING CLASS (0-14) ['No building','W1' 'W2' 'C1L' 'C1M' 'C1H' 'C2L' 'C2M' 'C2H' 'C3L' 'C3M''C3H' 'S1L' 'S1M' 'S1H']
% col 15 = Building Period
% col 16 = Displacement spectral ordinates (in)
% col 17-20 = Fragility_Structural_Precode (17=Slight, 18=Moderate, 19=Extensive, 20=Complete)
% col 21-24 = Fragility_Structural_Low     (21=Slight, 22=Moderate, 23=Extensive, 24=Complete)
% col 25-28 = Fragility_Structural_Moderate(25=Slight, 26=Moderate, 27=Extensive, 28=Complete)
% col 29-32 = Fragility_Structural_High    (29=Slight, 30=Moderate, 31=Extensive, 32=Complete)
% col 33-36 = Arrange the Damage State     (33=Slight, 34=Moderate, 35=Extensive, 36=Complete) 
%(summation( P(DS>ds)* RMV )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

note that the above columns are 0 indexed in python (e.g. col - 1)
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

sys.path.insert(0, '../../py_helpers')
import DC_helper as DCH
from h5_writer import h5_writer as h5wrt
from csv_writer import csv_writer as csvwrt

np.random.seed(1337)
np.seterr(divide = 'ignore')	# ignoring warning signs. np.log(0) results in "-inf". "-inf" in st.norm.cdf = 0

class building_damage_eq():
	def __init__(self):
		self.setup()
		self.run()

	# ~~~ user input up top ~~~
	def user_input_vals(self):
		self.n_sims = 100			# number of simulations
		self.retrofit_vals = 4		# originally set to 4
		self.write_h5 = False 		# True/False: write all output to h5 file

	def setup(self):
		self.user_input_vals()
		self.load_mat_files()
		self.initialize_variables()
		self.preallocate_space()

	def run(self):
		self.print_runinfo()
		for self.retrofit in range(self.retrofit_vals):
			print('retrofit: {}' .format(self.retrofit))
			self.Data[self.Data[:,8] < self.retrofit, 8] = self.retrofit
			for self.r in range(len(self.RT)):
				self.load_RTIMs()
				for self.i in range(len(self.Data)):
					if self.Data[self.i, 4] == 0:
						self.Data_0()
					elif self.Data[self.i, 4] == 1:
						self.Data_1(1)
						self.prob_collapse(self.BuildingClass[0])
					elif self.Data[self.i, 4] == 2:
						self.Data_1(2)
						self.prob_collapse(self.BuildingClass[1])
					elif (self.Data[self.i, 4] == 3) and (self.Data[self.i, 5]<4):
						self.Data_1(3)
						self.prob_collapse(self.BuildingClass[2])
					elif (self.Data[self.i, 4] == 3) and (3<self.Data[self.i, 5]<8):
						self.Data_1(4)
						self.prob_collapse(self.BuildingClass[3])
					elif (self.Data[self.i, 4] == 3) and (self.Data[self.i, 5]>8):
						self.Data_1(5)
						self.prob_collapse(self.BuildingClass[4])
					elif (self.Data[self.i, 4] == 4) and (self.Data[self.i, 5]<4):
						self.Data_1(6)
						self.prob_collapse(self.BuildingClass[5])
					elif (self.Data[self.i, 4] == 4) and (3<self.Data[self.i, 5]<8):
						self.Data_1(7)
						self.prob_collapse(self.BuildingClass[6])
					elif (self.Data[self.i, 4] == 4) and (self.Data[self.i, 5]>8):
						self.Data_1(8)
						self.prob_collapse(self.BuildingClass[7])
					elif (self.Data[self.i, 4] == 5) and (self.Data[self.i, 5]<4):
						self.Data_1(9)
						self.prob_collapse(self.BuildingClass[8])
					elif (self.Data[self.i, 4] == 5) and (3<self.Data[self.i, 5]<8):
						self.Data_1(10)
						self.prob_collapse(self.BuildingClass[9])
					elif (self.Data[self.i, 4] == 5) and (self.Data[self.i, 5]>8):
						self.Data_1(11)
						self.prob_collapse(self.BuildingClass[10])
					elif (self.Data[self.i, 4] == 6) and (self.Data[self.i, 5]<4):
						self.Data_1(12)
						self.prob_collapse(self.BuildingClass[11])
					elif (self.Data[self.i, 4] == 6) and (3<self.Data[self.i, 5]<8):
						self.Data_1(13)
						self.prob_collapse(self.BuildingClass[12])
					elif (self.Data[self.i, 4] == 6) and (self.Data[self.i, 5]>8):
						self.Data_1(14)
						self.prob_collapse(self.BuildingClass[13])
				self.GS_DamageState[self.retrofit, self.r] = self.Data[:,32:36]

			self.define_building_reptime()
			for self.r in range(len(self.RT)):
				timer = time.time()
				self.cases = self.Data[:,4]>0
				for self.i in range(self.n_sims):
					self.building_eval()

				print('elapsed time: {}' .format(time.time() - timer))
		

	# ~~~ secondary methods ~~~
	#  ~~ from self.setup() ~~

	def load_mat_files(self):
		self.Data = DCH.readmat('Data_Seaside_V1.mat', 'Data', dtype = 'array')
		self.Fragility_Structural_Hazus = DCH.readmat('Fragility_Structural_Hazus.mat', 'Fragility_Structural_Hazus', dtype = 'array')

	def initialize_variables(self):
		self.BuildingClass_ID = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) 
		self.BuildingClass = np.array(['W1', 'W2', 'C1L', 'C1M', 'C1H', 'C2L', 'C2M', 'C2H', 'C3L', 'C3M', 'C3H', 'S1L', 'S1M', 'S1H'])
		self.BuildingPeriod = np.array([0.35, 0.4, 0.4, 0.75, 1.45, 0.35, 0.56, 1.09, 0.35, 0.56, 1.09, 0.5, 1.08, 2.21])
		self.T = np.array([0.3, 0.4, 0.4, 0.75, 1.5, 0.3, 0.6, 1, 0.3, 0.6, 1, 0.5, 1, 2])  # IM computed at this periods;
		self.g=386.4	# in/s2
		self.n_lots = len(self.Data[:,0])	# number of tax lots
		self.n_DS = 5						# number of damage states
		self.damage_ratio = np.array([0, 0.005, 0.155, 0.555, 0.9])

		self.RT = np.array([100, 200, 250, 500, 1000, 2500, 5000, 10000])

		self.Data[np.where(self.Data[:,4] == 2.5)[0], 4] = 3
		self.ori_building_code = self.Data[:,8]
		new_cols = np.zeros(len(self.Data)*23).reshape(len(self.Data), 23)
		self.Data = np.hstack((self.Data, new_cols))
		
		self.setup_fragilities_from_hazus()

		self.outfilename = 'building_damage_eq'

	def preallocate_space(self):
		RT_size = len(self.RT)

		self.time_90p_restore_building = np.zeros(2*self.retrofit_vals*RT_size*self.n_sims).reshape(2, self.retrofit_vals, RT_size, self.n_sims)	# "2" is for "fast_vals"
		self.time_100p_restore_building = np.zeros(2*self.retrofit_vals*RT_size*self.n_sims).reshape(2, self.retrofit_vals, RT_size, self.n_sims)
		self.tot_rep_cost_building = np.zeros(1*self.retrofit_vals*RT_size*self.n_sims).reshape(1, self.retrofit_vals, RT_size, self.n_sims)
		self.frac_damaged_building = np.zeros(1*self.retrofit_vals*RT_size*self.n_sims).reshape(1, self.retrofit_vals, RT_size, self.n_sims)
		self.frac_habitable_building = np.zeros(1*self.retrofit_vals*RT_size*self.n_sims).reshape(1, self.retrofit_vals, RT_size, self.n_sims)
		self.GS_DamageState = np.zeros(self.retrofit_vals*RT_size*len(self.Data)*4).reshape(self.retrofit_vals, RT_size, len(self.Data), 4)
		
	#  ~~ from self.run() ~~ 
	def print_runinfo(self):
		print('starting building damage evaluation')
		print('\tn_sims: {}' .format(self.n_sims))
		print('\tretrofit_vals: {}' .format(self.retrofit_vals))
		print('\twrite h5: {}' .format(self.write_h5))

	def load_RTIMs(self):
		filestr = 'IM_RTP{}YR.mat' .format(self.RT[self.r])
		IM_RTP = DCH.readmat(filestr, 'IM_RTP', dtype = 'array')
		IM_Sa = IM_RTP[:,1:]
		self.Sd = (IM_Sa*self.T**2*self.g)/(3.14*3.14*4)

	def Data_0(self):
			self.Data[self.i,13]=0
			self.Data[self.i,14]=0
			self.Data[self.i,15]=0
			self.Data[self.i,16:20]=0
			self.Data[self.i,20:24]=0
			self.Data[self.i,24:28]=0
			self.Data[self.i,28:32]=0
			self.Data[self.i,32:36] = 0

	def Data_1(self, val):
			self.Data[self.i, 13] = val
			self.Data[self.i, 14] = self.BuildingPeriod[val-1]
			self.Data[self.i, 15] = self.Sd[self.i, val-1]

	def prob_collapse(self, key_val):
		# Probability of collapse from appropriate Fragility_Structural_PreCode   
		if self.Data[self.i, 8] == 0:
			self.Data[self.i, 16] = st.norm.cdf(np.log(self.Data[self.i, 15]), np.log(self.Fragility_Structural[key_val]['Parameter'][3,0]), self.Fragility_Structural[key_val]['Parameter'][3,1])
			self.Data[self.i, 17] = st.norm.cdf(np.log(self.Data[self.i, 15]), np.log(self.Fragility_Structural[key_val]['Parameter'][3,2]), self.Fragility_Structural[key_val]['Parameter'][3,3])
			self.Data[self.i, 18] = st.norm.cdf(np.log(self.Data[self.i, 15]), np.log(self.Fragility_Structural[key_val]['Parameter'][3,4]), self.Fragility_Structural[key_val]['Parameter'][3,5])
			self.Data[self.i, 19] = st.norm.cdf(np.log(self.Data[self.i, 15]), np.log(self.Fragility_Structural[key_val]['Parameter'][3,6]), self.Fragility_Structural[key_val]['Parameter'][3,7])

		# Probability of collapse from appropriate Fragility_Structural_LowCode  
		elif self.Data[self.i, 8] == 1:
			self.Data[self.i, 20] = st.norm.cdf(np.log(self.Data[self.i, 15]), np.log(self.Fragility_Structural[key_val]['Parameter'][2,0]), self.Fragility_Structural[key_val]['Parameter'][2,1])
			self.Data[self.i, 21] = st.norm.cdf(np.log(self.Data[self.i, 15]), np.log(self.Fragility_Structural[key_val]['Parameter'][2,2]), self.Fragility_Structural[key_val]['Parameter'][2,3])
			self.Data[self.i, 22] = st.norm.cdf(np.log(self.Data[self.i, 15]), np.log(self.Fragility_Structural[key_val]['Parameter'][2,4]), self.Fragility_Structural[key_val]['Parameter'][2,5])
			self.Data[self.i, 23] = st.norm.cdf(np.log(self.Data[self.i, 15]), np.log(self.Fragility_Structural[key_val]['Parameter'][2,6]), self.Fragility_Structural[key_val]['Parameter'][2,7])

		# Probability of collapse from appropriate Fragility_Structural_ModerateCode        
		elif self.Data[self.i, 8] == 2:
			self.Data[self.i, 24] = st.norm.cdf(np.log(self.Data[self.i, 15]), np.log(self.Fragility_Structural[key_val]['Parameter'][1,0]), self.Fragility_Structural[key_val]['Parameter'][1,1])
			self.Data[self.i, 25] = st.norm.cdf(np.log(self.Data[self.i, 15]), np.log(self.Fragility_Structural[key_val]['Parameter'][1,2]), self.Fragility_Structural[key_val]['Parameter'][1,3])
			self.Data[self.i, 26] = st.norm.cdf(np.log(self.Data[self.i, 15]), np.log(self.Fragility_Structural[key_val]['Parameter'][1,4]), self.Fragility_Structural[key_val]['Parameter'][1,5])
			self.Data[self.i, 27] = st.norm.cdf(np.log(self.Data[self.i, 15]), np.log(self.Fragility_Structural[key_val]['Parameter'][1,6]), self.Fragility_Structural[key_val]['Parameter'][1,7])	

		# Probability of collapse from appropriate Fragility_Structural_HighCode      
		elif self.Data[self.i, 8] == 3:
			self.Data[self.i, 28] = st.norm.cdf(np.log(self.Data[self.i, 15]), np.log(self.Fragility_Structural[key_val]['Parameter'][0,0]), self.Fragility_Structural[key_val]['Parameter'][0,1])
			self.Data[self.i, 29] = st.norm.cdf(np.log(self.Data[self.i, 15]), np.log(self.Fragility_Structural[key_val]['Parameter'][0,2]), self.Fragility_Structural[key_val]['Parameter'][0,3])
			self.Data[self.i, 30] = st.norm.cdf(np.log(self.Data[self.i, 15]), np.log(self.Fragility_Structural[key_val]['Parameter'][0,4]), self.Fragility_Structural[key_val]['Parameter'][0,5])
			self.Data[self.i, 31] = st.norm.cdf(np.log(self.Data[self.i, 15]), np.log(self.Fragility_Structural[key_val]['Parameter'][0,6]), self.Fragility_Structural[key_val]['Parameter'][0,7])	

		if self.Data[self.i, 8] == 0:
			self.Data[self.i,32] = self.Data[self.i,16] 
			self.Data[self.i,33] = self.Data[self.i,17] 
			self.Data[self.i,34] = self.Data[self.i,18] 
			self.Data[self.i,35] = self.Data[self.i,19]
		elif self.Data[self.i, 8] == 1:
			self.Data[self.i,32] = self.Data[self.i,20] 
			self.Data[self.i,33] = self.Data[self.i,21] 
			self.Data[self.i,34] = self.Data[self.i,22] 
			self.Data[self.i,35] = self.Data[self.i,23]
		elif self.Data[self.i, 8] == 2:
			self.Data[self.i,32] = self.Data[self.i,24] 
			self.Data[self.i,33] = self.Data[self.i,25] 
			self.Data[self.i,34] = self.Data[self.i,26] 
			self.Data[self.i,35] = self.Data[self.i,27]
		elif self.Data[self.i, 8] == 3:
			self.Data[self.i,32] = self.Data[self.i,28] 
			self.Data[self.i,33] = self.Data[self.i,29] 
			self.Data[self.i,34] = self.Data[self.i,30] 
			self.Data[self.i,35] = self.Data[self.i,31]

	def define_building_reptime(self):
		self.building_rep_time_mu = np.array([0.001*5, 0.5*120, 360, 720] )	# restoration time % table 15.10 and 15.11
		self.building_rep_time_std = np.array([0.5, 0.5, 0.5, 0.5])*self.building_rep_time_mu				# std for repair time
		self.building_rep_time_cov = self.building_rep_time_std/self.building_rep_time_mu	# COV of repair time
		self.building_rep_time_log_med = np.log(self.building_rep_time_mu/np.sqrt(self.building_rep_time_cov**2+1)) # lognormal parameters for repair time model
		self.building_rep_time_beta = np.sqrt(np.log(self.building_rep_time_cov**2+1))
		self.building_rep_time_covm = self.building_rep_time_beta[:, None]*self.building_rep_time_beta

		self.building_rep_time_mu2 = np.array([0.001*5, 0.5*120, 360, 720] )	# restoration time % table 15.10 and 15.11
		self.building_rep_time_std2 = np.array([0.5, 0.5, 0.5, 0.5])*self.building_rep_time_mu2				# std for repair time
		self.building_rep_time_cov2 = self.building_rep_time_std2/self.building_rep_time_mu2	# COV of repair time
		self.building_rep_time_log_med2 = np.log(self.building_rep_time_mu2/np.sqrt(self.building_rep_time_cov2**2+1)) # lognormal parameters for repair time model
		self.building_rep_time_beta2 = np.sqrt(np.log(self.building_rep_time_cov2**2+1))
		self.building_rep_time_covm2 = self.building_rep_time_beta2[:, None]*self.building_rep_time_beta2

	def building_eval(self):
		DS = 5-np.sum(np.heaviside(np.random.uniform(low=0, high=1, size=(self.n_lots,1)) - self.GS_DamageState[self.retrofit, self.r], 0.5), axis = 1) 	# damage state

		rep_time_building_all = np.column_stack((np.zeros(self.n_lots), np.exp(np.random.multivariate_normal(self.building_rep_time_log_med,self.building_rep_time_covm,self.n_lots))))	# generating correlated repair time estimates
		rep_time_building_all2 = np.column_stack((np.zeros(self.n_lots), np.exp(np.random.multivariate_normal(self.building_rep_time_log_med2,self.building_rep_time_covm2,self.n_lots))))	# generating correlated repair time estimates

		rep_time_buildings = np.array([rep_time_building_all[int(i), int(obj)-1] for i, obj in enumerate(DS)])
		rep_time_buildings2 = np.array([rep_time_building_all2[int(i), int(obj)-1] for i, obj in enumerate(DS)])

		self.time_90p_restore_building[0, self.retrofit, self.r, self.i] = np.percentile(rep_time_buildings[self.cases], 90)
		self.time_100p_restore_building[0, self.retrofit, self.r, self.i] = np.max(rep_time_building_all[self.cases])
		
		self.time_90p_restore_building[1, self.retrofit, self.r, self.i] = np.percentile(rep_time_buildings2[self.cases], 90)
		self.time_100p_restore_building[1, self.retrofit, self.r, self.i] = np.max(rep_time_building_all2[self.cases])
		
		rep_costs_buildings = self.damage_ratio[DS.astype(int)-1] * self.Data[:,10]
		self.tot_rep_cost_building[0, self.retrofit, self.r, self.i] = np.sum(rep_costs_buildings[self.cases])

		self.habitable = np.zeros(np.size(DS))
		self.habitable[DS==1] = 1
		self.habitable[DS==5] = 0

		cases1 = np.array(DS==2)
		temp = np.random.uniform(low=0, high=1, size = np.sum(cases1))
		self.habitable[cases1] = temp<=0.75

		cases1 = np.array(DS==3)
		temp = np.random.uniform(low=0, high=1, size = np.sum(cases1))
		self.habitable[cases1] = temp<=0.50

		cases1 = np.array(DS==4)
		temp = np.random.uniform(low=0, high=1, size = np.sum(cases1))
		self.habitable[cases1] = temp<=0.25

		self.frac_habitable_building[0, self.retrofit, self.r, self.i] = np.mean(self.habitable[self.cases])
		self.frac_damaged_building[0, self.retrofit, self.r, self.i] = np.mean(DS[self.cases]>2)

	# ~~~ tertiary and beyond methods ~~~
	def setup_fragilities_from_hazus(self):
		self.Fragility_Structural = {}
		for i, obj in enumerate(self.BuildingClass):
			self.Fragility_Structural[obj] = {}
			self.Fragility_Structural[obj]['ID'] = self.BuildingClass_ID[i]
			self.Fragility_Structural[obj]['Parameter'] = self.Fragility_Structural_Hazus[:, 8*i:8*(i+1)] 



bde = building_damage_eq()

if bde.__dict__['write_h5'] == True:
	h5wrt(bde)

