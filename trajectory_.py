import numpy as np

# initialization
selected_client = [0, 2, 4]
gradients_dict = {client_id: [] for client_id in selected_client} # from training
gram_matrix = True # argparse
ema_factor = 0.1 # argparse
ema_gram_matrices_dict = {client_id: [] for client_id in gradients_dict.keys()}

# velocity = grad at the current training round - grad at the previous training round 
def compute_velocity(gradients_dict):
    velocities_dict = {}
    
    for client_id, gradient_list in gradients_dict.items():
        if len(gradient_list) >= 2:
            velocities = [gradient_list[i] - gradient_list[i - 1] for i in range(1, len(gradient_list))]
            velocities_dict[client_id] = velocities
        else:
            velocities_dict[client_id] = []

    return velocities_dict

# form gram matrix
def gram_matrix(gradients_dict, velocities_dict):
    gram_matrices_dict = {}
    
    for client_id in gradients_dict:
        gradients = gradients_dict[client_id]
        velocities = velocities_dict.get(client_id, [])  # Get velocities for the client_id or an empty list if not available
        
        # Check if there are gradients and velocities for this client ID
        if gradients and velocities:
            gram_matrices = [
                np.outer(
                    np.concatenate((gradient, velocity)),
                    np.concatenate((gradient, velocity))
                )  # Gram Matrix
                for gradient, velocity in zip(gradients, velocities)
            ]
            gram_matrices_dict[client_id] = gram_matrices
        else:
            gram_matrices_dict[client_id] = []

    return gram_matrices_dict

# smoothing
def update_ema_gram_matrices(ema_gram_matrices_dict, gram_matrices_dict):
    for client_id in gram_matrices_dict.keys():
        new_gram_matrices = gram_matrices_dict[client_id]
        ema_gram_matrices = ema_gram_matrices_dict[client_id]

        if not ema_gram_matrices:
            ema_gram_matrices_dict[client_id] = new_gram_matrices
        else:
            for i in range(len(new_gram_matrices)):
                ema_gram_matrices[i] = ema_factor * new_gram_matrices[i] + (1 - ema_factor) * ema_gram_matrices[i]
    
    return ema_gram_matrices, ema_gram_matrices_dict

# alignment due to the diff num of local training rounds of each clients
def interpolate_gradient_histories(gradients_dict):
    # find the maximum history length
    max_length = max(len(history) for history in gradients_dict.values())

    # interpolate missing gradients
    interpolated_gradients_dict = {}
    for client_id, gradient_list in gradients_dict.items():
        interpolated_gradients = []

        for i in range(max_length):
            if i < len(gradient_list):
                # use known gradients if available
                interpolated_gradients.append(gradient_list[i])
            else:
                # interpolate missing gradients
                if len(gradient_list) >= 2:
                    prev_gradient = gradient_list[-2]
                    next_gradient = gradient_list[-1]
                    interpolation_factor = (i - len(gradient_list) + 1) / 1.0
                    interpolated_gradient = prev_gradient + interpolation_factor * (next_gradient - prev_gradient)
                    interpolated_gradients.append(interpolated_gradient)
                else:
                    # if there are fewer than 2 known gradients, cannot interpolate, so use zeros
                    interpolated_gradients.append(np.zeros_like(gradient_list[0]))

        interpolated_gradients_dict[client_id] = interpolated_gradients

    return interpolated_gradients_dict

def run_trajectory(gradients_dict, ema_gram_matrices_dict):
    if gram_matrix: # method 1: with Gram Matrix
        velocities_dict = compute_velocity(gradients_dict)
        gram_matrices_dict = gram_matrix(gradients_dict, velocities_dict)
        ema_gram_matrices_dict, gram_matrices_dict = update_ema_gram_matrices(ema_gram_matrices_dict, gram_matrices_dict)
        final_trajectory = interpolate_gradient_histories(ema_gram_matrices_dict)
    else: # method 2: withthout Gram Matrix
        ema_gram_matrices_dict, gram_matrices_dict = update_ema_gram_matrices(ema_gram_matrices_dict, gradients_dict)
        final_trajectory = interpolate_gradient_histories(ema_gram_matrices_dict)

    return ema_gram_matrices_dict, final_trajectory

