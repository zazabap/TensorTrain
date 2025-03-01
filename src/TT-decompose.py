import numpy as np
from scipy.linalg import svd

def tensor_train_decomposition(X, max_rank=10):
    """
    Compute the Tensor Train (TT) decomposition of a given tensor X.
    
    Parameters:
        X (ndarray): The input tensor of shape (d1, d2, ..., dN).
        max_rank (int): Maximum allowable rank (truncation parameter) for each SVD.
    
    Returns:
        cores (list): A list containing the TT-cores.
        ranks (list): A list of TT-ranks including boundary ranks.
    """
    # Get the shape and number of dimensions of the tensor
    X_shape = X.shape
    N = len(X_shape)
    
    # Start with boundary rank 1
    ranks = [1]
    cores = []
    
    # Reshape the tensor to prepare for the first SVD:
    # For the first mode, flatten all dimensions except the first.
    tensor = X.reshape(X_shape[0], -1)
    
    # Loop over each mode except the last one
    for k in range(N - 1):
        # Perform the SVD on the current reshaped tensor
        U, S, Vt = svd(tensor, full_matrices=False)
        
        # Determine the rank for truncation (should not exceed max_rank)
        r_k = min(len(S), max_rank)
        
        # Truncate the matrices to keep the most significant components
        U = U[:, :r_k]
        S = np.diag(S[:r_k])
        Vt = Vt[:r_k, :]
        
        # Reshape U into the current TT-core: shape (r_{k-1}, d_k, r_k)
        core_shape = (ranks[-1], X_shape[k], r_k)
        core = U.reshape(core_shape)
        cores.append(core)
        ranks.append(r_k)
        
        # Prepare the tensor for the next iteration: multiply S and Vt
        tensor = S @ Vt
        
        # Reshape the tensor: combine the current rank and the next mode dimension
        new_shape = (r_k * X_shape[k + 1], -1)
        tensor = tensor.reshape(new_shape)
    
    # The last core: reshape the remaining tensor into shape (r_{N-1}, d_N, 1)
    last_core = tensor.reshape(ranks[-1], X_shape[-1], 1)
    cores.append(last_core)
    
    return cores, ranks

# Example Usage:
# Create a random 3D tensor (for example, 4x4x4)
X = np.random.rand(4, 4, 4)
tt_cores, tt_ranks = tensor_train_decomposition(X, max_rank=2)

# Print the shapes of the TT-cores and the TT-ranks
print("TT-ranks:", tt_ranks)
for i, core in enumerate(tt_cores):
    print(f"Core {i+1} shape: {core.shape}")
