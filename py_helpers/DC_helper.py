import igraph as ig
import numpy as np
import scipy.stats as st
from scipy import io

def capacity_reptime_gen(frag_med, frag_covm, reptime_log_med, reptime_covm, n_links):
	capacity = np.exp(np.random.multivariate_normal(np.log(frag_med), frag_covm, n_links)) # generating correlated capacity estimates
	reptime = np.column_stack((np.zeros(n_links), np.exp(np.random.multivariate_normal(reptime_log_med,reptime_covm,n_links))))	# generating correlated repair time estimates
	return capacity, reptime

def capacity_reptime_gen_2d(frag_med, frag_covm, reptime_log_med, reptime_covm, n_links):
	capacity = np.array([np.exp(np.random.multivariate_normal(np.log(frag_med[ii]), frag_covm[:,:,ii])) for ii in range(n_links)]) # generating correlated capacity estimates for bridges. note that numpy's mv. norm function only takes a 1XN array for mean, hence the "loop"
	reptime = np.column_stack((np.zeros(n_links), np.exp(np.random.multivariate_normal(reptime_log_med, reptime_covm, n_links))))
	return capacity, reptime

def DS_eval(demand, capacity, closure_ds):
	DS = 1+np.sum(np.heaviside(demand[:,None] - capacity, 0.5), axis = 1) 	# damage state
	closed = DS>closure_ds	# where damage is greater than max_ds
	return closed, DS

def DS_eval_1d(demand, capacity, closure_ds):
	DS = 1+np.sum(np.heaviside(demand - capacity, 0.5), axis = 1) 	# damage state
	closed = DS>closure_ds
	return closed, DS

def adj_creator(adj_org, closed_ID, start_node, end_node):
	adj_post_EQ = [i[:] for i in adj_org]
	for n in range(len(closed_ID)):
		adj_post_EQ[start_node[closed_ID[n]]][end_node[closed_ID[n]]] = 0
		adj_post_EQ[end_node[closed_ID[n]]][start_node[closed_ID[n]]] = 0
	return adj_post_EQ

def create_graph(adj):
	return ig.Graph.Adjacency(adj, mode = 'undirected')

def delete_edges(G, start_node, end_node, closed_ID):
	edge_list = [(start_node[closed_ID[i]], end_node[closed_ID[i]]) for i in range(len(closed_ID))]
	G.delete_edges(edge_list)
	return G

def delete_edge_rowcol(G, start_node, end_node, closed_ID):
	e_from = G.es.select(_source_in = start_node[closed_ID])
	G.delete_edges(e_from)
	e_to = G.es.select(_target_in = start_node[closed_ID])
	G.delete_edges(e_to)
	return G

def conn_comp(G, n_nodes):
	"""
	converting above to appropriate format for rest of script.
	example:
		bins = [[4],[2],[0,1,3,5,6]]
		bins2 = [2,2,1,2,0,2,2]
	"""
	bins = list(G.components())
	bins2 = np.zeros(n_nodes)
	for ii, obj in enumerate(bins):
		for iii in obj:
			bins2[iii] = ii
	return bins2

def readmat(matfile, key, dtype, idx_0 = False):
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