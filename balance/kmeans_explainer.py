import numpy as np


class CFE_explainer:
	def __init__(self, m_source, m_target):
		"""
		Initialization method.

		Parameters:
		- m_source (np.ndarray): The centroid of the current cluster.
		- m_target (np.ndarray): The centroid of the target cluster.
		"""

		self.m_source = np.asarray(m_source, dtype=np.float64)
		self.m_target = np.asarray(m_target, dtype=np.float64)

		if not (m_source.shape == m_target.shape):
			raise ValueError("m_source and m_target must have the same shape.")

		self.center_distance = np.linalg.norm(self.m_target - self.m_source) ** 2

	def compute_counterfactual(self, y, mask, d_eps_center_dist_ratio):
		"""
		Compute the counterfactual for a given point y based on the centroids m_source and m_target.
		
		Parameters:
		- y (np.ndarray): The original point.
		- mask (np.ndarray): A binary mask array of the same shape as y where:
							 - 1 indicates the element can be modified (free).
							 - 0 indicates the element is fixed and cannot be modified.
		- d_eps_center_dist_ratio (float): Distance from the bountary relative to cluster centers dinstance.
		
		Returns:
		- z (np.ndarray): The actionable counterfactual point.
		"""
		
		# Ensure inputs are numpy arrays
		y = np.asarray(y, dtype=np.float64)
		mask = np.asarray(mask, dtype=np.int32)

		# Input validation
		if not (y.shape == self.m_source.shape == self.m_target.shape == mask.shape):
			raise ValueError("All input arrays must have the same shape.")

		if not (0 <= d_eps_center_dist_ratio and d_eps_center_dist_ratio <= 1):
			raise ValueError("d_eps_center_dist_ratio must be a value between 0 and 1.")

		d_eps = self.center_distance * d_eps_center_dist_ratio
		
		# Compute the boundary hyperplane parameters
		v = self.m_source - self.m_target
		c = (np.dot(self.m_source, self.m_source) - np.dot(self.m_target, self.m_target) - d_eps) / 2 
		
		# Split y, v based on the mask into free and fixed components
		y_free = y[mask == 1]
		y_fixed = y[mask == 0]
		v_free = v[mask == 1]
		v_fixed = v[mask == 0]
		
		# Calculate the modified constraint
		constraint_value = c - np.dot(y_fixed, v_fixed)
		
		# Compute the projection of y_free onto the hyperplane
		if np.dot(v_free, v_free) == 0: return None  
		
		# Solution for the free components of z
		z_free = y_free - ((np.dot(y_free, v_free) - constraint_value) / np.dot(v_free, v_free)) * v_free

		# Combine the free and fixed components to form z
		z = np.copy(y)
		z[mask == 1] = z_free  # Replace free components
	 
		return z

	def compute_mahalanobis_counterfactual(self, y, R, d_eps_center_dist_ratio=0.0):
		"""
		Compute the counterfactual minimizing the Mahalanobis distance with k-means cluster constraint,
		allowing adjustment relative to the hyperplane using d_eps.

		Parameters:
		- y (np.ndarray): The factual point.
		- R (np.ndarray): The positive definite matrix for Mahalanobis distance.
		- d_eps_center_dist_ratio (float): Distance from the boundary relative to cluster centers' distance.

		Returns:
		- z (np.ndarray): The optimal counterfactual point.
		"""
		y = np.asarray(y, dtype=np.float64)
		R = np.asarray(R, dtype=np.float64)

		# Input validation
		if y.shape != self.m_source.shape:
			raise ValueError("y must have the same shape as m_source and m_target.")
		if R.shape[0] != R.shape[1] or R.shape[0] != y.shape[0]:
			raise ValueError("R must be a square matrix with dimensions matching y.")
		if not (0 <= d_eps_center_dist_ratio <= 1):
			raise ValueError("d_eps_center_dist_ratio must be a value between 0 and 1.")

		# Compute d_eps based on center distance and ratio
		d_eps = self.center_distance * d_eps_center_dist_ratio

		# Compute the necessary parameters
		delta_m = self.m_target - self.m_source  # Difference vector
		numerator = (
			2 * delta_m.T @ y 
			- (np.dot(self.m_target, self.m_target) - np.dot(self.m_source, self.m_source) + d_eps)
		)
		denominator = delta_m.T @ R @ delta_m
		lambda_ = numerator / denominator

		# Compute the counterfactual point z*
		z = y - (lambda_ / 2) * (R @ delta_m)
		return z
