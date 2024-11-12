# implementation of the proposed algorithm - NEURAL

# importing the libraries
import igraph
import networkx as nx
import copy
import random
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
import time

# function to create an adjacency list for the graph
def get_adj_list(E):
	Adjacency_List = {}
	for i in range (0, len(E)):
		e = E[i]
		s = e[0]
		t = e[1]
		if (s in Adjacency_List.keys()):
			Adjacency_List[s].append(t)
		else:
			Adjacency_List[s] = []
			Adjacency_List[s].append(t)
		if (t in Adjacency_List.keys()):
			Adjacency_List[t].append(s)
		else:
			Adjacency_List[t] = []
			Adjacency_List[t].append(s)
	return Adjacency_List

# get external connections for target community nodes
def ext_conn(comm, target, IG_edgeList):
  con = []
  for i in comm:
    count = 0
    for j in target:
      if ((i, j) in IG_edgeList or (j, i) in IG_edgeList):
        count = count + 1
    con.append(count)
  return con

# main method for NEURAL
def perm_loss_decep(target_comm, IG_edgeList, deg, in_deg, e_max, comm_max, communities, subedge, subgraph, subvertices, beta, target_comm_index):

	add_loss = 0
	del_loss = 0
	while (True):
    
		add_node, loss, max_comm = get_max_loss_node(target_comm, deg[target_comm_index], in_deg[target_comm_index], e_max[target_comm_index], comm_max[target_comm_index])
		con_list = ext_conn(communities[max_comm], target_comm, IG_edgeList)
		add_node_2, loss_2, max_comm_2 = get_max_loss_node_2(communities[max_comm], deg[max_comm], in_deg[max_comm], e_max[max_comm], comm_max[max_comm], con_list)
		add_loss = loss + loss_2
		best_edges = getBestDelExclBridges(target_comm, subedge, subgraph, subvertices)
		((del_node, del_node_2), loss) = get_del_max_loss(target_comm, best_edges, deg[target_comm_index], in_deg[target_comm_index], e_max[target_comm_index], subgraph)
		del_loss = loss
		
		if add_loss >= del_loss and add_loss > 0:
			if ((add_node, add_node_2) not in IG_edgeList) and ((add_node_2, add_node) not in IG_edgeList):
			  IG_edgeList.append((add_node, add_node_2))
			  ind_node = target_comm.index(add_node)
			  deg[target_comm_index][ind_node] = deg[target_comm_index][ind_node] + 1
			  e_max[target_comm_index][ind_node] = e_max[target_comm_index][ind_node] + 1


			  ind_node_2 = communities[max_comm].index(add_node_2)
			  deg[max_comm][ind_node_2] = deg[max_comm][ind_node_2] + 1
			  e_max[max_comm][ind_node_2] = e_max[max_comm][ind_node_2] + 1

		elif del_loss > 0:
			IG_edgeList.remove((del_node, del_node_2))
			ind_node = target_comm.index(del_node)
			deg[target_comm_index][ind_node] = deg[target_comm_index][ind_node] - 1
			in_deg[target_comm_index][ind_node] = in_deg[target_comm_index][ind_node] - 1
			ind_node = target_comm.index(del_node_2)
			deg[target_comm_index][ind_node] = deg[target_comm_index][ind_node] - 1
			in_deg[target_comm_index][ind_node] = in_deg[target_comm_index][ind_node] - 1
			subedge.remove((del_node, del_node_2))
			subgraph[del_node].remove(del_node_2)
			subgraph[del_node_2].remove(del_node)

		beta = beta - 1
		if (beta > 0 and (add_loss > 0 or del_loss > 0)):
			continue
		else:
			break
	return IG_edgeList

# get target community node that would bring the maximum permanence loss in case of inter-community edge addition
def get_max_loss_node(target_comm, deg, in_deg, e_max, comm_max):

	max_loss = 0
	node = target_comm[0]
	max_comm = 0
	if len(target_comm) == 1:
		node = target_comm[0]
	for i in range (0, len(target_comm)):
		if (e_max[i] == 0):
			loss_orig = 1/(deg[i])
		else:
			loss_orig = 1/(deg[i]*e_max[i])
		loss_new = 1/((deg[i] + 1)*(e_max[i] + 1))
		loss = (loss_orig - loss_new)*in_deg[i]
		if loss > max_loss:
			max_loss = loss
			node = target_comm[i]
			max_comm = comm_max[i]
	return node, max_loss, max_comm

# get target community node that would bring the maximum permanence loss in case of inter-community edge addition
def get_max_loss_node_2(target_comm, deg, in_deg, e_max, comm_max, con_list):

	max_loss = 0
	node = -1
	max_comm = 0
	if len(target_comm) == 1:
		node = target_comm[0]
	for i in range (0, len(target_comm)):
		if (e_max[i] == 0):
			loss_orig = 1/(deg[i])
		else:
			loss_orig = 1/(deg[i]*e_max[i])
		if (con_list[i] == e_max[i]):
			loss_new = 1/((deg[i] + 1)*(e_max[i] + 1))
		else:
			if (e_max[i] == 0):
				loss_new = 1/(deg[i] + 1)
			else:
				loss_new = 1/((deg[i] + 1)*e_max[i])
		loss = (loss_orig - loss_new)*in_deg[i]
		if loss > max_loss:
			max_loss = loss
			node = target_comm[i]
			max_comm = comm_max[i]
	return node, max_loss, max_comm

# find a node in the E_max(u) community to add an inter-community edge
def get_add_node(target_comm, IG_edgeList, communities, add_node, comm_max):

	ind_node = target_comm.index(add_node)
	ind_comm = comm_max[ind_node]
	comm = communities[ind_comm]
	node = -1
	for i in comm:
		if ((i, add_node) not in IG_edgeList and (add_node, i) not in IG_edgeList):
			node = i
	return node

# find the local clustering coefficient for a node
def get_c_in(node, subgraph):

	node_neighbours = subgraph[node]
	num = len(node_neighbours)
	count = 0
	for i in node_neighbours:
		li = subgraph[i]
		for j in li:
			if j in node_neighbours:
				count = count + 1
	if (num*(num - 1) == 0):
		ratio = (count/2)/((num)/2)
	else:
		ratio = (count/2)/((num*(num - 1))/2)
	return ratio

# get the non bridging edges in the target community
def getBestDelExclBridges(target_comm, edges, Adjacency_List, num_vertices):
	
	best_edges = []
	for i in edges:
		Cpy_Adj_List = copy.deepcopy(Adjacency_List)
		Cpy_Adj_List[i[0]].remove(i[1])
		Cpy_Adj_List[i[1]].remove(i[0])
		try:
			if(connectedComponents(target_comm, num_vertices, Cpy_Adj_List)) == 1:
				best_edges.append(i)
		except:
			continue
	return best_edges

# calculating the number of components for the subgraph spanned by vertices of target community (used for finding the bridge edges)
def DFSUtil(target_comm, temp, v, visited, Adjacency_List):

	visited[v] = True
	temp.append(v)
	for i in Adjacency_List[target_comm[v]]:
		if visited[target_comm.index(i)] == False:
			temp = DFSUtil(target_comm, temp, target_comm.index(i), visited, Adjacency_List)
	return temp

def connectedComponents(target_comm, num_vertices, Adjacency_List):
	visited = [] 
	cc = [] 
	for i in range(num_vertices):
		visited.append(False)
	for v in range(num_vertices):
		if visited[v] == False: 
			temp = [] 
			cc.append(DFSUtil(target_comm, temp, v, visited, Adjacency_List))
	return len(cc)

# get the intra-community edge to be deleted
def get_del_max_loss(target_comm, best_edges, deg, in_deg, e_max, subgraph):

	max_loss = 0
	node_u = 0
	node_v = 0
	for i in best_edges:
		u = target_comm.index(i[0])
		v = target_comm.index(i[1])
		if (e_max[u] != 0):
			u_loss_1 = (1/e_max[u])*((deg[u] - in_deg[u])/(deg[u]*(deg[u] - 1)))
		else:
			err = (deg[u]*(deg[u] - 1))
			if (err == 0):
				err = 1
			u_loss_1 = ((deg[u] - in_deg[u])/(err))
		if (e_max[v] != 0):
			v_loss_1 = (1/e_max[v])*((deg[v] - in_deg[v])/(deg[v]*(deg[v] - 1)))
		else:
			err = (deg[u]*(deg[u] - 1))
			if (err == 0):
				err = 1
			v_loss_1 = ((deg[v] - in_deg[v])/(err))
		u_loss_2_a = get_c_in(i[0], subgraph)
		v_loss_2_a = get_c_in(i[1], subgraph)
		subgraph_prime = copy.deepcopy(subgraph)
		subgraph_prime[i[0]].remove(i[1])
		subgraph_prime[i[1]].remove(i[0])
		u_loss_2_b = get_c_in(i[0], subgraph_prime)
		v_loss_2_b = get_c_in(i[1], subgraph_prime)
		loss = u_loss_1 + v_loss_1 + (u_loss_2_a - u_loss_2_b) + (v_loss_2_a - v_loss_2_b)
		if loss > max_loss:
			max_loss = loss
			node_u = i[0]
			node_v = i[1]

	return ((node_u, node_v), max_loss)

def get_random_comm(communities, target_comm):

  max_len = 0
  index = 0
  for i in range (0, len(communities)):
    if (communities[i] != target_comm):
      len_ = len(communities[i])
      if len_ > max_len:
        max_len = len_
        index = i
  return index

def num_comm(target_comm, communities):
  uni_comm = []
  comm_list = []
  for node in target_comm:
    for c in communities:
      if node in c:
        comm_list.append(c)
        if c not in uni_comm:
          uni_comm.append(c)
          break
  return len(uni_comm), comm_list

def get_targetComm_Neighbours(target_comm, communities, Adjacency_List):
	List = []
	marked = dict()
	for i in target_comm:
		for j in Adjacency_List[i]:
			if j not in marked:
				for k in range(len(communities)):
					if j in communities[k]:
						List.append(k)
						marked[j] = j
	return List, marked

def check_neighbours(neighbours, communities):
	ctr = 0
	List = []
	for i in range(len(communities)):
		for j in communities[i]:
			if j in neighbours:
				List.append(i)
				ctr += 1
			if ctr == len(neighbours):
				return List
	return List

def get_entropy(labels, base = None):
  values, counts = np.unique(labels, return_counts = True)
  return entropy(counts, base = base)

# pass the network for which the algorithm has to be run
start_time = time.process_time()
path = "./Datasets/pol.gml"
nov = 105
graph = nx.read_gml(path, label = "id")   

e_ = list(graph.edges) 

Adjacency_List = get_adj_list(e_)

# specify nodes in the network
num_vertices = nov

IG_edgeList = []

for i in e_:
  IG_edgeList.append((i[0], i[1]))

g = igraph.Graph(directed = False)
g.add_vertices(num_vertices)
g.add_edges(IG_edgeList)

# get the community structure for the network passing the network through community detection algorithms
communities = g.community_multilevel()
comm_1 = copy.deepcopy(communities)
safe_copy_comm = copy.deepcopy(communities)

comm_length = len(communities)
  
NMI_List = []
Neighbourhood_NMI_List = []
sum_comm = 0
sum_entropy = 0
print(len(communities))
# run over all target communities
for i in range (0, len(communities)):
  print(i)
  graph = nx.read_gml(path, label = "id")
  e_ = list(graph.edges)
  Adjacency_List = get_adj_list(e_)
  num_vertices = nov

  IG_edgeList = []
  for j in e_:
    IG_edgeList.append((j[0], j[1]))

  g = igraph.Graph(directed = False)
  g.add_vertices(num_vertices)
  g.add_edges(IG_edgeList)

  target_comm = communities[i]
  target_comm_index = i
  pre_neighbours, neighbours = get_targetComm_Neighbours(target_comm, comm_1, Adjacency_List)

  # calculating different network properties

  deg = []
  for c in communities:
    deg_list = []
    for j in c:
      deg_list.append(g.vs[j].degree())
    deg.append(deg_list)

  in_deg = []
  for c in communities:
    in_deg_list = []
    for j in c:
      in_ = 0
      for k in Adjacency_List[j]:
        if k in c:
          in_ = in_ + 1
      in_deg_list.append(in_)
    in_deg.append(in_deg_list)

  e_max_list = []
  comm_max_list = []
  for c in communities:
    e_max = []
    comm_max = []
    for l in c:
      max_count = 0
      comm = -1
      for j in range (0, len(communities)):
        count = 0
        if communities[j] != c:
          for k in Adjacency_List[l]:
            if k in communities[j]:
              count = count + 1
        if count > max_count:
          max_count = count
          comm = j
      if comm == -1:
        index = get_random_comm(communities, target_comm)
        comm = index
      e_max.append(max_count)
      comm_max.append(comm)
    e_max_list.append(e_max)
    comm_max_list.append(comm_max)

  subgraph = {}
  for l in Adjacency_List.keys():
    if l in target_comm:
      subgraph[l] = []
      for j in Adjacency_List[l]:
        if j in target_comm:
          subgraph[l].append(j)

  subedge = []
  for l in IG_edgeList:
    if l[0] in target_comm and l[1] in target_comm:
      subedge.append(l)

  subvertices = len(target_comm)

  # selecting a value for budget beta
  beta = int(0.3*len(target_comm))
  IG_edgeList_ = perm_loss_decep(target_comm, IG_edgeList, deg, in_deg, e_max_list, comm_max_list, communities, subedge, subgraph, subvertices, beta, target_comm_index)

  # communities in the updated graph
  g = igraph.Graph(directed = False)
  num_vertices = nov
  g.add_vertices(num_vertices)
  g.add_edges(IG_edgeList_)

  communities = g.community_multilevel()
  post_neighbours = check_neighbours(neighbours, communities)
  
  num_splits, comm_list = num_comm(target_comm, communities)
  sum_comm = sum_comm + num_splits
  
  nmi = igraph.compare_communities(comm_1, communities, method = "nmi")
  
  nmi_neighbourhood = igraph.compare_communities(pre_neighbours, post_neighbours, method = "nmi")
  
  entropy_val = get_entropy(comm_list)
  sum_entropy = sum_entropy + entropy_val

  NMI_List.append(nmi)
  Neighbourhood_NMI_List.append(nmi_neighbourhood)
  communities = safe_copy_comm

end_time = time.process_time()
print(f"NMI list: {NMI_List}")
# print(f"Neighboors NMI list: {Neighbourhood_NMI_List}")
# print(f"Sum entropy: {sum_entropy}, Sum comm: {sum_comm}")
print(f"Time cost: {end_time - start_time}")
