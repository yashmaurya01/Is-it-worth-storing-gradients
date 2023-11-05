import numpy as np
import torch
import itertools

N = 10 # num_clients at current round

# smooth with EMA
def smooth_client_gradients(historical_grads, client_grads, beta):
    smoothed_grads = {}
    for client_id, grads in client_grads.items():
        smoothed_grads[client_id] = {}
        for name, grad in grads.items():
            historical_grad = historical_grads[client_id][name]
            updated_grad = beta * historical_grad + (1 - beta) * grad
            historical_grads[client_id][name] = updated_grad
            smoothed_grads[client_id][name] = updated_grad

    return smoothed_grads

# stat of each neuron's historical gradients
def gradient_statistics_neuron(historical_grads):
    stats = {}
    for client_id, client_historical_grads in historical_grads.items():
        stats[client_id] = {}
        for param_name, gradients in client_historical_grads.items():
            # Assuming gradients are 2D (for fully connected layers) or 4D (for convolutional layers)
            # Adjust the following line if your layers have a different structure
            for neuron_idx, neuron_grad in enumerate(gradients.view(-1, gradients.size(-1))):
                grad_array = neuron_grad.detach().cpu().numpy()
                neuron_stats = {
                    'mean': np.mean(grad_array),
                    'max': np.max(grad_array),
                    'min': np.min(grad_array),
                    'std': np.std(grad_array),
                    '95th': np.percentile(grad_array, 95),
                    '85th': np.percentile(grad_array, 85),
                    '75th': np.percentile(grad_array, 75),
                    '50th': np.percentile(grad_array, 50),  # Median
                    '25th': np.percentile(grad_array, 25),
                    '15th': np.percentile(grad_array, 15),
                    '5th': np.percentile(grad_array, 5),
                }

                stats[client_id][(param_name, neuron_idx)] = neuron_stats

    return stats

# stat of each layer's historical gradients
def gradient_statistics_layer(historical_grads):
    stats = {}
    for client_id, client_historical_grads in historical_grads.items():
        stats[client_id] = {}
        for layer_name, layer_grads in client_historical_grads.items():
            grad_array = layer_grads.detach().cpu().numpy().flatten()
            layer_stats = {
                'mean': np.mean(grad_array),
                'max': np.max(grad_array),
                'min': np.min(grad_array),
                'std': np.std(grad_array),
                '95th': np.percentile(grad_array, 95),
                '85th': np.percentile(grad_array, 85),
                '75th': np.percentile(grad_array, 75),
                '50th': np.percentile(grad_array, 50),  # Median
                '25th': np.percentile(grad_array, 25),
                '15th': np.percentile(grad_array, 15),
                '5th': np.percentile(grad_array, 5),
            }

            stats[client_id][layer_name] = layer_stats

    return stats

# cos similarity between clients
def cos_similarity_matrix(gradients):
    cosine_similarities = torch.zeros((N, N))

    for i in range(N):
        for j in range(i, N):  # Start from i to avoid redundant calculations
            if i == j:
                cosine_similarities[i, j] = 1
            else:
                similarity = torch.nn.functional.cosine_similarity(
                    gradients[i].flatten(),
                    gradients[j].flatten(),
                    dim=0, eps=1e-10)
                cosine_similarities[i, j] = similarity
                cosine_similarities[j, i] = similarity
    
    return cosine_similarities


