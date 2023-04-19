''' Author : Raaed Syed
'''

import sys
import numpy as np
import cv2
from scipy.spatial.distance import cdist

class KMeans():
    def __init__(self, image, k = 5, out = '',  max_iters = 1000):
        self.k = k 
        self.max_iters = max_iters
        self.output = out

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.original_image = image

        image = image.reshape((-1, 3))
        float_image = np.float32(image)
        self.image = float_image
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = [] 

    def predict(self):
        pixels, features = self.image.shape
        
        # initializing random centroids 
        random_idx = np.random.choice(pixels, self.k, replace = False)
        self.centroids = [self.image[idx] for idx in random_idx]

        # K means algorithm - assign each pixel to a cluster + update centroids
        for i in range(self.max_iters):
            # assign each point to a cluster
            self.clusters = self.create_clusters(self.centroids)

            # update the cluster centers 
            old_centroids = self.centroids
            self.centroids = self.update_centroids(self.clusters)

            if self.has_converged(old_centroids, self.centroids):
                print(f'converged in {i} iterations')
                break

        output_image = self.pixel_values(self.centroids, self.clusters)
        cv2.imwrite(self.output, output_image)

    def create_clusters(self, centroids):      
        distances = cdist(self.image, centroids, metric = 'cityblock')
        clusters = np.argmin(distances, axis=1)

        curr_clusters = [np.where(clusters == i)[0] for i in range(self.k)]
        return curr_clusters

    def update_centroids(self, clusters):
        centroids = np.array([np.mean(self.image[cluster], axis=0) for cluster in clusters])
        return centroids
    
    def has_converged(self, old_cent, cent):
        return np.array_equal(old_cent, cent)

    def pixel_values(self, centroids, clusters):
        num_pixels = self.image.shape[0]
        labels = np.zeros(num_pixels)

        for cluster_id, cluster in enumerate(clusters):
            for pix_id in cluster:
                labels[pix_id] = cluster_id

        labels = labels.astype(int)

        centroids = np.uint8(centroids)

        output_image = centroids[labels]
        output_image = output_image.reshape(self.original_image.shape)

        return output_image

if __name__ == '__main__':
    
    if len(sys.argv) < 3:
        print('Usage: python KMeans <input-image> <k> <output-image>')

    try: 
        image = cv2.imread(sys.argv[1])
        k = int(sys.argv[2])
        output = sys.argv[3]

        km = KMeans(image = image, k = k, out = output)
        km.predict() 

    except Exception as e: 
        print(e)