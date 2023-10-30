import numpy as np
from sklearn.metrics import pairwise_kernels
from sklearn.svm import SVC
from scipy.optimize import curve_fit

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# selected clients
selected_client = [0, 2, 4]

# trajectory
gradients_dict = {client_id: [] for client_id in selected_client}

# velocity = grad at the current training round - grad at the previous training round 
def compute_velocity(gradients_dict):
    velocities_dict = {
        client_id: [
            current_gradient - previous_gradient # velocity
            for current_gradient, previous_gradient in zip(
                gradients_dict[client_id][1:], gradients_dict[client_id][:-1]
            )
        ]
        for client_id in gradients_dict
    }
    return velocities_dict

# form gram matrix
def gram_matrix(gradients_dict, velocities_dict):
    gram_matrices_dict = {
        client_id: [
            np.outer(
                np.concatenate((gradient, velocity)),
                np.concatenate((gradient, velocity))
            )  # Gram Matrix
            for gradient, velocity in zip(gradients_dict[client_id], velocities_dict[client_id])
        ]
        for client_id in gradients_dict
    }
    return gram_matrices_dict


# smoothing: fit the diagonal of each Gram matrix to an exponential decay curve
# ?? will this erase the sudden change in the trajectory caused by an attack?
# ?? perhaps under non-IID setting, smoothing works well since the trajectory may act as a trend rather than a spike
# ?? maybe can try through experiment
def smooth_trajectory(gram_matrices_dict):
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    smoothed_dict = {client_id: [curve_fit(func, range(len(gram_matrix)), gram_matrix.diagonal()) 
                                 for gram_matrix in gram_matrices_dict[client_id]] 
                     for client_id in gram_matrices_dict}
    return smoothed_dict

# since the length of the trajectory for each client is different, we need to perform alignment
# gak: A kernel for time series based on global alignments
def global_alignment_kernel(smoothed_dict):
    gak_dict = {client_id: pairwise_kernels(smoothed_dict[client_id], metric='laplacian') 
                for client_id in smoothed_dict}
    return gak_dict

# ?? should we better use classification or regression
def classify_Kmeans(all_data):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(all_data)
    labels = kmeans.labels_


# data preprocessing
velocities_dict = compute_velocity(gradients_dict)
gram_matrices_dict = gram_matrix(gradients_dict, velocities_dict)
smoothed_dict = smooth_trajectory(gram_matrices_dict)
gak_dict = global_alignment_kernel(smoothed_dict)

# classification
all_data = np.vstack([np.array(gak_dict[client_id]) for client_id in gak_dict])
classifier = classify_Kmeans(all_data)
