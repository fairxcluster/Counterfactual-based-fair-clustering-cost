import numpy as np
import random
import time
from math import gcd
import networkx as nx
import matplotlib.pyplot as plt
from MWD_utils import MWDistance 

class VanillaFairletDecomposition(object):
	"""
	Computes a vanilla fairlet decomposition that ensures fair clusters. It might not give the optimal cost value.
	"""

	def __init__(self, p, q, blues, reds, data, weights, means, covariances):
		"""
		p (int) : First balance parameter 
		q (int) : Second balance parameter
		blues (list) : Index of the points corresponding to first class
		reds (list) : Index of the points corresponding to second class
		data (list) : Contains actual data points
		"""
		self.p = p
		self.q = q
		self.blues = blues
		self.reds = reds
		self.data = data
		self.weights = weights
		self.means = means
		self.covariances = covariances  

	def balanced(self, r, b):
		"""
		Checks for initial balance and feasibility.

		Args:
			r (int) : Total length of majority class
			b (int) : Total length of minority class

		Returns:
			bool value indicating whether the balance is possible
		"""
		if r == 0 and b == 0:
			return True
		if r == 0 or b == 0:
			return False
		return min(float(r / b), float(b / r)) >= float(self.p / self.q)
 
	def make_fairlet(self, points, dataset, fairlets, fairlet_centers, costs):
		"""
		Adds fairlet to the fairlet decomposition and returns the sum of MWD distances from fairlet members to center.
		
		Args:
			points (list) : Index of the points that comprise the fairlet
			dataset (list) : Original data
			fairlets (list)
			fairlet_centers (list)
			costs (list)
		"""
		cost_list = [sum([MWDistance(dataset[center], dataset[point], self.weights, self.means, self.covariances) for point in points]) for center in points]
		cost, center = min((cost, center) for (center, cost) in enumerate(cost_list))
		fairlets.append(points)
		fairlet_centers.append(points[center])
		costs.append(cost)

		return
	 
	def decompose(self):
		"""
		Computes vanilla (p , q) - fairlet decomposition of given points as per Lemma 3 in NeurIPS 2017 paper.
		Assumes that balance parameters are non-negative integers such that gcd(p, q) = 1.
		Also assumes that balance of reds and blues is atleast p/q.

		Returns:
			fairlets (list) 
			fairlet_centers (list)
			costs (list)
		"""
		assert gcd(self.p, self.q) == 1, 'Please ensure that the GCD of balance parameters is 1.'
		assert self.p <= self.q, 'Please use balance parameters such that p <= q.'
		
		fairlets = []
		fairlet_centers = []
		fairlet_costs = []
		
		if len(self.reds) < len(self.blues): # We want the reds to be bigger in size as they correspond to 'q' parameter
			temp = self.blues
			self.blues = self.reds
			self.reds = temp
			
		R = len(self.reds)
		B = len(self.blues)
		
		assert self.balanced(R, B), 'Input sets are unbalanced: ' + str(R) + ' , ' + str(B)
	 
		# If both reds and blues are empty, return empty results
		if R == 0 and B == 0:
			return fairlets, fairlet_centers, fairlet_costs
	 
		b = 0
		r = 0
		
			
		while ((R - r) - (B - b)) >= (self.q - self.p) and (R - r) >= self.q and (B - b) >= self.p:
			self.make_fairlet(self.reds[r: (r + self.q)] + self.blues[b: (b + self.p)], self.data, fairlets, fairlet_centers, fairlet_costs)
			r += self.q
			b += self.p
		if ((R - r) + (B - b)) >= 1 and ((R - r) + (B - b)) <= (self.p + self.q):
			self.make_fairlet(self.reds[r:] + self.blues[b:], self.data, fairlets, fairlet_centers, fairlet_costs)
			r = R
			b = B
		elif ((R - r) != (B - b)) and ((B - b) >= self.p):
			self.make_fairlet(self.reds[r: r + (R - r) - (B - b) + self.p] + self.blues[b: (b + self.p)], self.data, fairlets,
									 fairlet_centers, fairlet_costs)
			r += (R - r) - (B - b) + self.p
			b += self.p
		assert (R - r) == (B - b), 'Error in computing fairlet decomposition.'
		for i in range(R - r):    
			self.make_fairlet([self.reds[r + i], self.blues[b + i]], self.data, fairlets, fairlet_centers, fairlet_costs)

		print("%d fairlets have been identified."%(len(fairlet_centers)))
		assert len(fairlets) == len(fairlet_centers)
		assert len(fairlet_centers) == len(fairlet_costs)
		
		return fairlets, fairlet_centers, fairlet_costs

class MCFFairletDecomposition(object):
	"""
	Computes the optimized version of fairlet decomposition for a Gaussian Mixture Model using minimum-cost flow.
	The decomposition is based on the model-weighted distance (MWD) between points.
	"""

	def __init__(self, blues, reds, t, distance_threshold, data, weights, means, covariances):
		"""
		blues (list) : Index of the points corresponding to first class
		reds (list) : Index of the points corresponding to second class
		t (int) : (1, t) is the fairness ratio to be enforced
		distance_threshold (int) : Remnant from the k-center implementation. Not used in our implementation
		data (list) : Contains actual data points
		"""
		self.blues = blues
		self.blue_nodes = len(blues)
		self.reds = reds
		self.red_nodes = len(reds)
		self.weights = weights
		self.means = means
		self.covariances = covariances        
       
		assert self.blue_nodes >= self.red_nodes

		self.t = t
		self.distance_threshold = distance_threshold
		self.data = data

		# Initializing the Graph
		self.G = nx.DiGraph()

	def compute_distances(self):
		"""
		Compute MWD distances between every pair of blue and red nodes.
		"""

		self.distances = {}
		for idx, i in enumerate(self.blues):
			for idx2, j in enumerate(self.reds):
				self.distances['B_%d_R_%d'%(idx+1, idx2+1)] = MWDistance(self.data[i], self.data[j],self.weights,self.means,self.covariances)

	def build_graph(self, plot_graph=False, weight_limit=10000000):
		"""
		Builds the graph i.e. nodes and edges.

		Args:
			plot_graph (bool) : Indicates whether the graph needs to be plotted
			weight_limit (int) : Big value to be used in place of infinity for cost definition
		"""

		self.G.add_node('beta', pos=(0, 4+(1+max(self.blue_nodes, self.red_nodes))/2), demand=(-1*self.red_nodes))
		self.G.add_node('ro', pos=(5, 4+(1+max(self.blue_nodes, self.red_nodes))/2), demand=(self.blue_nodes))
		self.G.add_edge('beta', 'ro', weight=0, capacity=min(self.blue_nodes, self.red_nodes))

		for i in range(self.blue_nodes):
			self.G.add_node('B%d'%(i+1), pos=(1, i+1), demand=-1)
			self.G.add_edge('beta', 'B%d'%(i+1), weight=0, capacity=self.t-1) #self.t-1
		for i in range(self.red_nodes):
			self.G.add_node('R%d'%(i+1), pos=(4, i+1), demand=1)
			self.G.add_edge('R%d'%(i+1), 'ro', weight=0, capacity=self.t-1) #self.t-1
			
		# Latent nodes
		for i in range(self.blue_nodes):
			for j in range(self.t):
				position = (i+1) + ((i+1 - i) / self.t)*j
				self.G.add_node('B%d_%d'%(i+1, j+1), pos=(2, position), demand=0)
				self.G.add_edge('B%d'%(i+1), 'B%d_%d'%(i+1, j+1), weight=0, capacity=1)
		for i in range(self.red_nodes):
			for j in range(self.t):
				position = (i+1) + ((i+1 - i) / self.t)*j
				self.G.add_node('R%d_%d'%(i+1, j+1), pos=(3, position), demand=0)
				self.G.add_edge('R%d_%d'%(i+1, j+1), 'R%d'%(i+1), weight=0, capacity=1)
				
		# Adding edges between latent nodes
		for i in range(self.blue_nodes):
			for j in range(self.t):
				for k in range(self.red_nodes):
					for l in range(self.t):
						dist_raw = self.distances['B_%d_R_%d'%(i+1, k+1)]
						dist = int(np.round(dist_raw,2)*100) # Algorithm cant take floats as weights in graph. 
						self.G.add_edge('B%d_%d'%(i+1, j+1), 'R%d_%d'%(k+1, l+1), weight=dist, capacity=1) # Add the distances on the edges    
                     
		if plot_graph:
			if self.blue_nodes > 10:
				print("Graph can't be plotted because the blue nodes exceed 10.")
				print("Beginning GMM MCF Decomposition")    
			else:
				plt.figure(figsize=(10, 8))
				pos = {n : (x, y) for (n, (x, y)) in nx.get_node_attributes(self.G, 'pos').items()}
				nx.draw_networkx_nodes(self.G, pos, node_size=1000, alpha=0.5)
				nx.draw_networkx_labels(self.G, pos, font_size=11)
				nx.draw_networkx_edges(self.G, pos)
				plt.show()

	def decompose(self):
		"""
		Calls the network simplex to run the MCF algorithm.
		Computes the fairlets and fairlet centers.
		The fairlet centers returned by the function are found by minimizing a kmedian cost with MWD.
		The cost returned is also the kmedian cost with MWD distances.
		We change the fairlet centers to the mean of the fairlet members in the experiment code in notebooks
		Bank, Diabetes and Census. The cost of the GMM MWD decomposition is also computed in those notebooks
		(and for the final results) as the sum of MWD distances to the center (the mean) of the fairlets. 

		Returns:
			fairlets (list) 
			fairlet_centers (list)
			costs (list)
		"""

		start_time = time.time()
		flow_cost, flow_dict = nx.network_simplex(self.G)
		print("Time taken to compute MCF solution - %.3f seconds."%(time.time() - start_time))

		fairlets = {}
		# Assumes mapping from blue nodes to the red nodes
		points_seen = set() # keep track of points that have been added to fairlets
		for i in flow_dict.keys():
			if 'B' in i and '_' in i:
				if sum(flow_dict[i].values()) == 1:
					for j in flow_dict[i].keys():
						if flow_dict[i][j] == 1:
							point = i.split('_')[0]
							if point not in points_seen: # check if point has already been added to a fairlet
								points_seen.add(point) # add point to set of seen points
								if j.split('_')[0] not in fairlets:
									fairlets[j.split('_')[0]] = [point]
								else:
									fairlets[j.split('_')[0]].append(point)
				
		fairlets = [([a] + b) for a, b in fairlets.items()]

		fairlets2 = []
		for i in fairlets:
			curr_fairlet = []
			for j in i:
				if 'R' in j:
					d = self.reds
				else:
					d = self.blues
				curr_fairlet.append(d[int(j[1:]) - 1])
			fairlets2.append(curr_fairlet)
		fairlets = fairlets2
		del fairlets2

		# Choosing fairlet centers
		fairlet_centers = []
		fairlet_costs = []
		for points in fairlets:   
			cost_list = [sum([MWDistance(self.data[center], self.data[point], self.weights, self.means, self.covariances) for point in points]) for center in points]
			cost, center = min((cost, center) for (center, cost) in enumerate(cost_list))
            
			fairlet_centers.append(points[center])
			fairlet_costs.append(cost)
            
		# Flattening fairlet list
		flat_list = []
		for i in range(len(fairlets)):
			for j in range(len(fairlets[i])):
				flat_list.append(fairlets[i][j])
		flat_list = np.sort(flat_list)

		# Checking if all points have been assigned
		missing_points = []
		for i in range(len(self.data)):
			if (i in list(flat_list)) == False:
				missing_points.append(i)

		# Finding size 2 fairlets for non assigned points to be assigned to
		s2 = [] # fairlets of size 2 
		for i in range(len(fairlets)):
			if len(fairlets[i])==2:
				s2.append(i)       
            
		# Looking at distances to other fairlets to assign potentially missing points
		for j in range(len(missing_points)):
			dist = []
			ids = []
			for i in s2:
				dist.append((MWDistance(self.data[missing_points[j]],self.data[fairlet_centers[i]],self.weights,self.means,self.covariances)))
				ids.append(i)

			assigned_fairlet_id = ids[np.argmin(dist)] # the assigned fairlet id 
			fairlets[assigned_fairlet_id].append(missing_points[j]) # assigning the point
			pop_item = np.where(np.array(s2)==assigned_fairlet_id)[0][0] # deleting the now used fairlet from the search list
			s2.pop(pop_item)
			fairlet_costs[assigned_fairlet_id] += np.min(dist) # adding the cost of the extra point

		print("%d fairlets have been identified."%(len(fairlet_centers)))
		assert len(fairlets) == len(fairlet_centers)
		assert len(fairlet_centers) == len(fairlet_costs)

		return fairlets, fairlet_centers, fairlet_costs