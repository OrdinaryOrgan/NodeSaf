from networkx.algorithms.community import LFR_benchmark_graph
import pickle

num_vertices = 10000
LFR_graph = LFR_benchmark_graph(num_vertices, 2, 1.5, 0.4, average_degree = 20, min_degree = None, max_degree = 100, min_community = None, max_community = None, tol = 1e-07, max_iters = 500, seed = None)
filename = "graph"
outfile = open(filename, 'wb')
pickle.dump(LFR_graph, outfile)
outfile.close()