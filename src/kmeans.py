'''
This module will provide the core implementation of the K-Means algorithm
If we intend to publish this, we will need to ensure the algorithm is optimized
and fast. Use numpy for fast math operations
'''
import random

import numpy as np
from sklearn import preprocessing 

class KMeans:
    def __init__(self, 
                 n_clusters=3, 
                 tolerance=0.001, 
                 max_iterations=200, 
                 random_seed=42):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        random.seed(random_seed)

    def get_computed_centroids():
        return self.centroids

    '''
    Run KMeans on data vector
    data (np.ndarray) must have a shape of: 
        (N_SAMPLES, N_FEATURES)
    
    '''
    def fit(self, data):
        if len(data.shape) != 2:
            print('data must have shape (N_SAMPLES, N_FEATURES)')

        # Get input data number of data points and number of dimensions
        n_samples = data.shape[0]
        n_features = data.shape[1]
        print(f'Num samples: {n_samples}')
        print(f'Num features: {n_features}')

        # Normalize input data
        data_normalized = preprocessing.normalize(data)

        # ----- Initialize cluster centers -----
        min_val = np.amin(data_normalized)
        max_val = np.amax(data_normalized)

        self.centroids = np.ndarray(shape=(self.n_clusters, n_features))
        for i in range(self.n_clusters):
            self.centroids[i] = np.random.uniform(min_val, max_val, n_features)

        # Create array to store which cluster the data at each index is in
        self.cluster_allocations = np.ndarray(shape=(n_samples), dtype=int)

        # Main Loop
        for iteration in range(self.max_iterations):

            # ----- Assign points to clusters -----
            for data_idx, features in enumerate(data_normalized):
                centroid_distances = np.ndarray(shape=(self.n_clusters), dtype=float)
                
                for j, centroid in enumerate(self.centroids):
                    centroid_distances[j] = np.linalg.norm(features - centroid)

                self.cluster_allocations[data_idx] = np.argmin(centroid_distances)

            # ----- Re-calculate centroids -----    
            previous = self.centroids.copy()
            
            for i in range(self.n_clusters):
                cluster_data = []
                for data_idx, cluster in enumerate(self.cluster_allocations):
                    if cluster == i:
                        cluster_data.append(data_normalized[data_idx])

                if len(cluster_data) == 0:
                    random_index = np.random.randint(
                        len(self.cluster_allocations)
                    )
                    self.cluster_allocations[random_index] = i
                    cluster_data.append(data_normalized[random_index])

                self.centroids[i] = np.mean(np.array(cluster_data), axis=0)             
            
            print(f'\nCentroids for iteration {iteration}:\n {self.centroids}')

            # ----- Check for convergance -----
            isConverged = True

            for centroid in range(self.n_clusters):
                prev_centroid = previous[centroid] 
                cur_centroid = self.centroids[centroid]

                delta = abs(np.sum((cur_centroid - prev_centroid) / prev_centroid * 100.0))
                
                if delta > self.tolerance:
                    isConverged = False

                print(f'Change in centroid {centroid}: {delta}')

            if isConverged:
                break
            