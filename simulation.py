import numpy as np

# Define parameters
num_devices = 5
num_classes = 10
num_samples_per_device = 100
num_rounds = 3
threshold = 0.05  # Adjustable


def compute_loss(predictions, ground_truth):
    return np.mean((predictions - ground_truth)**2)

losses = []
device_data = [np.random.rand(num_samples_per_device, num_classes) for _ in range(num_devices)]
#Label poisoning attack: flipped(if binary) or uniformly random noisy labels
def label_poisoning(data, flip_prob):
    flip_mask = np.random.rand(*data.shape) < flip_prob
    # print("flip_mask",flip_mask)
    data[flip_mask] = 1 - data[flip_mask]
    return data

#Random updates attack: sending random zero-mean gaussian mparameters
def random_updates(data, mean=0, std=1):
    update = np.random.normal(mean, std, data.shape)
    return data + update

#Model replacement attack : scaling adversarial updates to make them dominate the update
def model_replacement(data, scale):
    return data * scale

# Simulate communication rounds with attacks
for round in range(num_rounds):
    print(f"Round {round+1}:")
    
    # Apply label poisoning attack (A1)
    for i in range(num_devices):
        device_data[i] = label_poisoning(device_data[i], flip_prob=0.2)
    
    # Apply random updates attack (A2)
    for i in range(1, num_devices):
        device_data[i] = random_updates(device_data[i], mean=0, std=0.1)  # Apply A2 to other devices
    
    # Apply model replacement attack (A3)
    for i in range(num_devices):
        if round > 0:  # Apply A3 from the second round onwards
            device_data[i] = model_replacement(device_data[i], scale=2)

    # Aggregate data and perform global training (not implemented yet)
    
    # Compute loss after each round
    # Note to self: Replace the following line with your actual prediction and ground truth data
    # For demonstration purposes, I'm using random data
    predictions = np.random.rand(num_samples_per_device, num_classes)
    ground_truth = np.random.rand(num_samples_per_device, num_classes)
    loss = compute_loss(predictions, ground_truth)
    
    losses.append(loss)
    
    # Detect adversarial attack
    if round > 0 and abs(loss - losses[round-1]) > threshold:
        print("Potential adversarial attack detected!")
    else:
        print("No likelihood of adversarial attack.")



