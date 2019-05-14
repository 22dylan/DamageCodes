"""
-converts IDU coverage from Envision to the appropriate numpy matrix for
use in building damage codes. 
-input includes:
	-idu_path: path to idu coverage (in .dbf format)
	-event: an event name (e.g. M90_SA00)
-both input values are specified as strings. 

dylan r. sanderson
feb. 15th, 2018
oregon state university
"""

import os
import numpy as np
import dbfread as dbf
import pandas as pd
import scipy as sp

class IDU_to_py():
	def __init__(self, idu_path, event):
		print('\nconverting IDU coverage')
		self.read_idu(idu_path, event)
		self.idu_to_array()
		self.read_ground_shaking()

	def read_idu(self, idu_path, event):
		data = dbf.DBF(idu_path, encoding='utf-8')
		self.frame = pd.DataFrame(iter(data))
		self.sa_vals = np.column_stack((self.frame[event + '_SA00'], self.frame[event + '_SA03'], self.frame[event + '_SA10']))
		
	def idu_to_array(self):
		self.Data = np.zeros(len(self.frame)*36).reshape(len(self.frame), 36)

		# self.Data[:,0] = self.frame['']
		# self.Data[:,1] = self.frame['']
		self.Data[:,2] = self.frame['IDU_INDEX']
		self.Data[:,3] = self.frame['yearbuilt']
		
		ind = np.where((self.frame['prop_ind'] == 'AGRICULTURAL') & (self.frame['landuse'] == 'FARMS'))[0]
		self.Data[ind,4] = 1 # assuming farm buildings are W1 --NOTE this will be overwritten if the total value = land value - i.e. no nuilding present in the parcel
		self.Data[ind,5] = 1 # single story

		ind = np.where((self.frame['prop_ind'] == 'AGRICULTURAL') & (self.frame['landuse'] == 'FOREST'))[0]
		self.Data[ind,4] = 0 # assuming no buildings in forest parcels
		self.Data[ind,5] = 0 # 

		ind = np.where((self.frame['prop_ind'] == 'AGRICULTURAL') & (self.frame['landuse'] == 'FISHERIES'))[0]
		self.Data[ind,4] = 0 # assuming no buildings in forest parcels
		self.Data[ind,5] = 0 

		ind = np.where((self.frame['prop_ind'] == 'AGRICULTURAL') & (self.frame['landuse'] == 'FIELD & SEED') & (self.frame['yearbuilt'] > 0))[0]
		self.Data[ind,4] = 1 # assuming farm buildings to be W1
		self.Data[ind,5] = 1 

		ind = np.where((self.frame['prop_ind'] == 'AGRICULTURAL') & (self.frame['landuse'] == 'FIELD & SEED') & (self.frame['yearbuilt'] == 0))[0]
		self.Data[ind,4] = 0 # if there is no building or no data then assigning 0
		self.Data[ind,5] = 0 

		ind = np.where(self.frame['prop_ind'] == 'SINGLE FAMILY RESIDENCE')[0]
		self.Data[ind,4] = 1 # assuming that single family residential buildings are single story wood buildings - W1
		self.Data[ind,5] = 1 

		ind = np.where((self.frame['prop_ind'] == 'SINGLE FAMILY RESIDENCE') & (self.frame['bedrooms']>=4))[0] # SFR with more than 3 bedrooms is assumed to be a 2 stroy wood building		self.Data[ind,4] = 2 # assuming that single family residential buildings are single story wood buildings - W2
		self.Data[ind,4] = 2 
		self.Data[ind,5] = 2

		ind = np.where((self.frame['prop_ind'] == 'APARTMENT') & (self.frame['bedrooms']>=4))[0] # SFR with more than 3 bedrooms is assumed to be a 2 stroy wood building
		self.Data[ind,4] = 2 # assuming that single family residential buildings are single story wood buildings - W2
		self.Data[ind,5] = 2 

		ind = np.where(self.frame['prop_ind'] == 'PARKING')[0]
		self.Data[ind,4] = 3 # assuming that parking structures are concrete buildings
		self.Data[ind,5] = 4 

		ind = np.where(self.frame['prop_ind'] == 'AMUSEMENT-RECREATION')[0]
		self.Data[ind,4] = 6 # assuming that AMUSEMENT-RECREATION buildings are steel moment frame buildings
		self.Data[ind,5] = 4 

		ind = np.where(self.frame['prop_ind'] =='COMMERCIAL BUILDING')[0]
		self.Data[ind,4] = 3 # C1 for commercial
		self.Data[ind,5] = 3 

		ind = np.where(self.frame['prop_ind'] == 'COMMERCIAL CONDOMINIUM')[0]
		self.Data[ind,4] = 3 # C1 for commercial condos
		self.Data[ind,5] = 4 

		ind = np.where(self.frame['prop_ind'] == 'CONDOMINIUM')[0]
		self.Data[ind,4] = 1 # W1 for condos
		self.Data[ind,5] = 1 

		ind = np.where((self.frame['prop_ind'] == 'CONDOMINIUM') & (self.frame['bedrooms']>=4))[0] # Condos with more than 3 bedrooms are assumed to be a 2 stroy wood building
		self.Data[ind,4] = 2 # assuming that single family residential buildings are single story wood buildings - W2
		self.Data[ind,5] = 2 

		ind = np.where(self.frame['prop_ind'] == 'DUPLEX')[0]
		self.Data[ind,4] = 2 # W2 for duplex
		self.Data[ind,5] = 2 

		ind = np.where(self.frame['prop_ind'] == 'FINANCIAL INSTITUTION')[0]
		self.Data[ind,4] = 5 # C3
		self.Data[ind,5] = 2 

		ind = np.where(self.frame['prop_ind'] == 'HOTEL, MOTEL')[0]
		self.Data[ind,4] = 2 # these are mostly frat houses
		self.Data[ind,5] = 2 

		ind = np.where((self.frame['prop_ind'] == 'HOTEL, MOTEL') &  (self.frame['landuse'] == 'HOTEL'))[0]
		self.Data[ind,4] = 3 # C1 
		self.Data[ind,5] = 5 

		ind = np.where((self.frame['prop_ind'] == 'HOTEL, MOTEL') & (self.frame['landuse'] == 'MOTEL'))[0]
		self.Data[ind,4] = 2 # W2 
		self.Data[ind,5] = 2 

		ind = np.where(self.frame['prop_ind'] == 'INDUSTRIAL')[0]
		self.Data[ind,4] = 6 # S1 
		self.Data[ind,5] = 1 

		ind = np.where(self.frame['prop_ind'] == 'INDUSTRIAL HEAVY')[0]
		self.Data[ind,4] = 6 # S1 
		self.Data[ind,5] = 1 

		ind = np.where(self.frame['prop_ind'] == 'OFFICE BUILDING')[0]
		self.Data[ind,4] = 5 # C3 
		self.Data[ind,5] = 3 

		ind = np.where(self.frame['prop_ind'] == 'RETAIL')[0]
		self.Data[ind,4] = 5 # C3 
		self.Data[ind,5] = 3 

		ind = np.where((self.frame['prop_ind'] == 'RETAIL') & (self.frame['landuse'] == 'FOOD STORES'))[0]
		self.Data[ind,4] = 5 # C3 
		self.Data[ind,5] = 1 

		ind = np.where((self.frame['prop_ind'] == 'RETAIL') & (self.frame['landuse'] == 'SHOPPING CENTER'))[0]
		self.Data[ind,4] = 6 # S1 
		self.Data[ind,5] = 3 

		ind = np.where((self.frame['prop_ind'] == 'RETAIL') & (self.frame['landuse'] == 'SUPERMARKET'))[0]
		self.Data[ind,4] = 6 # S1 
		self.Data[ind,5] = 1 

		ind = np.where((self.frame['prop_ind'] == 'RETAIL') & (self.frame['landuse'] == 'STORE BUILDING'))[0]
		self.Data[ind,4] = 5 # C3
		self.Data[ind,5] = 1 

		ind = np.where((self.frame['prop_ind'] == 'RETAIL') & (self.frame['landuse'] == 'AUTO SALES'))[0]
		self.Data[ind,4] = 6 # S1
		self.Data[ind,5] = 3 

		ind = np.where(self.frame['prop_ind'] == 'SERVICE')[0]
		self.Data[ind,4] = 4 # C2 
		self.Data[ind,5] = 3 

		ind = np.where((self.frame['prop_ind'] == 'SERVICE') & (self.frame['landuse'] == 'SCHOOL'))[0]
		self.Data[ind,4] = 4 # C2 
		self.Data[ind,5] = 3 

		ind = np.where((self.frame['prop_ind'] == 'SERVICE') & (self.frame['landuse'] == 'PUBLIC SCHOOL'))[0]
		self.Data[ind,4] = 4 # C2 
		self.Data[ind,5] = 3 

		ind = np.where((self.frame['prop_ind'] == 'SERVICE') & (self.frame['landuse'] == 'BAR'))[0]
		self.Data[ind,4] = 5 # C3
		self.Data[ind,5] = 2 

		ind = np.where((self.frame['prop_ind'] == 'SERVICE') & (self.frame['landuse'] == 'RESTAURANT BUILDING'))[0]
		self.Data[ind,4] = 5 # C3
		self.Data[ind,5] = 1 

		ind = np.where((self.frame['prop_ind'] == 'SERVICE') & (self.frame['landuse'] == 'NIGHTCLUB'))[0]
		self.Data[ind,4] = 5 # C3
		self.Data[ind,5] = 2 

		ind = np.where((self.frame['prop_ind'] == 'SERVICE') & (self.frame['landuse'] == 'LAUNDROMAT'))[0]
		self.Data[ind,4] = 6 # S1
		self.Data[ind,5] = 1 

		ind = np.where((self.frame['prop_ind'] == 'SERVICE') & (self.frame['landuse'] == 'FAST FOOD FRANCHISE'))[0]
		self.Data[ind,4] = 5 # C3
		self.Data[ind,5] = 1 

		ind = np.where((self.frame['prop_ind'] == 'SERVICE') & (self.frame['landuse'] == 'AUTO REPAIR'))[0]
		self.Data[ind,4] = 3 # C1
		self.Data[ind,5] = 2 

		ind = np.where((self.frame['prop_ind'] == 'SERVICE') & (self.frame['landuse'] == 'CARWASH'))[0]
		self.Data[ind,4] = 6 # S1
		self.Data[ind,5] = 1 

		ind = np.where((self.frame['prop_ind'] == 'SERVICE') & (self.frame['landuse'] == 'FUNERAL HOME'))[0]
		self.Data[ind,4] = 2 # W2
		self.Data[ind,5] = 2 

		ind = np.where((self.frame['prop_ind'] == 'SERVICE') & (self.frame['landuse'] == 'SERVICE STATION'))[0]
		self.Data[ind,4] = 3 # C1
		self.Data[ind,5] = 3 

		ind = np.where(self.frame['prop_ind'] == 'TRANSPORT')[0]
		self.Data[ind,4] = 0 # Neglecting ports and harbors 

		ind = np.where(self.frame['prop_ind'] == 'UTILITIES')[0]
		self.Data[ind,4] = 0 # Neglecting utility buildings

		ind = np.where(self.frame['prop_ind'] == 'WAREHOUSE')[0]
		self.Data[ind,4] = 6 # S1
		self.Data[ind,5] = 1 

		ind = (self.frame['prop_ind'] == self.frame['land_value'])[0]	# identifying vavant lots - can also use 'prop_ind' and identify rows with value 'VACANT'
		self.Data[ind, 4] = 0	# the data matrix is already already initialized with zeros. But still implementing this line for clarity and copleteness

		# self.Data[:,6] = self.frame['']
		# self.Data[:,7] = self.frame['']
		ind = np.where(self.Data[:,3] == 0)
		self.Data[ind, 8] = 0	# for cases where no data is available, assuming pre-1979 (drs, assumption)

		ind = np.where((self.Data[:,3] > 0) & (self.Data[:,3] < 1979))
		self.Data[ind, 8] = 0
		
		idx_low = np.where((self.Data[:,3] >= 1979) & (self.Data[:,3] < 1998))
		self.Data[ind, 8] = 1
		
		idx_mod = np.where((self.Data[:,3] >= 1998) & (self.Data[:,3] < 2008))
		self.Data[ind, 8] = 2
		
		idx_high = np.where(self.Data[:,3] >= 2008)
		self.Data[ind, 8] = 3
		
		self.Data[:,9] = self.frame['tot_value']
		self.Data[:,10] = self.frame['impr_value']
		self.Data[:,11] = self.frame['land_value']
		self.Data[:,12] = self.frame['sqfeet_liv']


	def read_ground_shaking(self):
		"""
		%This function interpolates Sa values at different periods using Sa at 0.0,
		%0.3, and 1.0 s assuming site class B for the Oregon Coast - following the
		%ASCE 7-10 response spectra shape

		% Assiming that sa_vals has the Sa values at 0.0, 0.3 and 1.0s (in this particular order)

		% assumed site class = B --> Fa = 1.0 and Fv = 1.0
		% Ss = 1.25g based on ASCE 7-10 figure 22.1
		% S1 = 0.6g

		written by Sabarethinam Kameshwar
		converted from matlab to python by dylan r. sanderson
		Feb. 2019
		"""
		n_idu = len(self.sa_vals)	#number of idus

		# Using the above values:
		T0 = 0.1 	# in s
		Ts = 0.5 	# in s
		Tl = 16 	# in s -- based on ASCE 7-10 figure 22.12
		T = np.array([0.3, 0.4, 0.4, 0.75, 1.5, 0.3, 0.6, 1, 0.3, 0.6, 1.0, 0.5, 1.0, 2.0])  # IM computed at this periods;
		Sa = np.zeros(n_idu*len(T)).reshape(n_idu, len(T))

		ind = ((T>=0) & (T<=T0))
		if np.sum(ind) != 0:
			Sa[:,ind] = self.sa_vals[:,0] + T[ind]*((self.sa_vals[:,2] - self.sa_vals[:,1])/(T0-0.0))

		ind = (T>T0) & (T<=Ts)
		if np.sum(ind) != 0:
			Sa[:,ind] = np.tile(self.sa_vals[:,1], (np.sum(ind), 1) ).transpose()

		ind = (T==1)
		if np.sum(ind) != 0:
			Sa[:,ind] = np.tile(self.sa_vals[:,2], (np.sum(ind), 1) ).transpose()

		ind = (T>Ts) & (T!=1)
		if np.sum(ind) != 0:
			f = sp.interpolate.PchipInterpolator([Ts, 1, Tl], [self.sa_vals[:,1], self.sa_vals[:,2], self.sa_vals[:,1]/16])
			Sa[:,ind] = f(T[ind]).transpose()

		self.IM_Sa = Sa/981 	# converting sa values to multiples of g

