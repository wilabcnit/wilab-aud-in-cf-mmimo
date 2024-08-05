"""
%% ----------- Generate Dataset ---------------
% Written by Muhammad Usman Khan
% Email id: muhammadusman.khan8@unibo.it
% Supervisor: Prof. Marco Chiani
% Department of Electrical, Electronic, and Information Engineering
% "Guglielmo Marconi" - DEI
% University of Bologna, Cesena Campus, Italy
% Version 1.0
% Created On: Feb 19, 2024
% Modified On:
% Description: Code for generating dataset for tranining, validtaion, and testing a CNN
 for active user detection in CF-mMIMO.
"""

import numpy as np
import pandas as pd
import sys
import os

def generate_samples(N_samples, L, D, M, K, N, S, D_p, tau, std_shadowing, var_h,var_n,eta):    
    """
    Function to generate samples for a dataset.
    
    Parameters:
    - N_samples: Number of samples to generate.
    - L: Path loss constant.
    - D: Length of the square area in meters.
    - M: Number of access points (APs).
    - K: Number of users.
    - N: Number of antennas at each AP.
    - S: Preamble sequence matrix.
    - D_p: Diagonal matrix with transmit power levels.
    - tau: Length of the preamble sequence.
    - std_shadowing: Standard deviation for shadowing.
    - var_h: Variance for small scale fading.
    - var_n: Variance for noise.
    - eta: Probability of a user being active.

    Returns:
    - samples_feat: Feature matrix for the samples.
    - samples_out: Output labels for the samples.
    """
    
    norm_values = np.sum(np.abs(S)**2, axis = 0)
    samples_feat = np.zeros((N_samples, M*N*K))
    samples_out = np.zeros((N_samples, K))
    
    for i in range(N_samples):
        # Generating the positions of APs within the square area
        dx = np.random.rand(M) * D
        dy = np.random.rand(M) * D
        
        # Generating the positions of users within the square area
        cod_K = np.random.rand(K, 2) * D
        
        # Calculating the distance between each user and AP
        dist = np.sqrt((cod_K[:, 0, np.newaxis] - dx)**2 + (cod_K[:, 1, np.newaxis] - dy)**2)
    
        # Path loss calculation
        PL = L + 23.0*np.log10(dist)
        
        # Shadowing effect
        shadowing = np.random.normal(0, std_shadowing, size=(K, M))
        shadowing = 10 ** (shadowing / 10.0)
        beta_sqrt = np.sqrt(10 ** (-PL / 10.0) * shadowing)
        
        
        # Generating user activity matrix (labels)
        labels = np.random.binomial(1, eta, K)
        D_a = np.diag(labels)
        
        # Channel gain coefficients for small scale fading
        h = (np.random.normal(size=(M,K,N)) + 1j*np.random.normal(size=(M,K,N)))*np.sqrt(var_h/2.0)
        G_m = np.repeat(beta_sqrt.flatten(), N).reshape(((M,K,N)))*h
      
        # Genrating Noise
        noise = (np.random.normal(size=(M,tau,N)) + 1j*np.random.normal(size=(M,tau,N)))*np.sqrt(var_n/2.0)
        
        # Final matrix calculations
        mat1 = np.matmul(S, D_a) # tau x K = (tau x K) x (K x K) 
        mat2 = np.matmul(mat1, D_p) # tau x K = (tau x K) x (K x K)
        mat3 = np.repeat(mat2[np.newaxis, ...], M, axis=0) #(M x tau x K)        
        mat4 = np.matmul(mat3, G_m) + noise # M x tau x N = (M x tau x K) x (M x K x N)
        mat5 = np.transpose(mat4, (0, 2, 1))
        mat6 = np.abs(np.matmul(mat5, np.conjugate(S))) 
        mat7 = mat6/norm_values  
        
        # Flatten the feature matrix and store it
        feat= mat7.flatten(order='C')
        samples_feat[i, :] = feat #np.concatenate([np.real(feat), np.imag(feat)])
        samples_out[i, :] = labels
    
    return samples_feat, samples_out

def main():
    # Parameters for the dataset generation
    D = 1000  # Length of the square area
    M = int(sys.argv[1]) # Number of access points
    K = int(sys.argv[2]) # Number of users
    N = 3; # Number of antennas at the AP
    tau = 40; # Length of the preamle sequence
    rho_bar = 200 # Transmit power
    noise_power = 1.2589*10**-11 # Noise power
    var_h = 1.0 # Variance for small Scale fading
    var_n = 1.0 # Variance for noise
    eta = 0.1 # Probability of a user being active
    testing_samples = 10**4 # Number of testing samples
    f = 1.9  # Carrier frequency in GHz
    std_shadowing = 5.9 # Standard deviation for shadowing

    # Variables
    rho = rho_bar/noise_power # Signal-to-noise ratio
    L = 32.4 + 20.0*np.log10(f) # Path loss constant
    
    
    # Read the Pilots/Preamble Sequnces from pkl file
    pilot_file = "pilots" + str(M) + ".pkl"
    pilots_feat = ["feature" + str(i) for i in range(K)]  #for output labels
    data = pd.read_pickle(pilot_file) 
    S = data[pilots_feat]
    S = S.to_numpy()
    real, imag = np.vsplit(S, 2)
    S = real + 1j*imag
    
    
    # Transmit Power
    D_p = np.zeros((K,K), float)
    np.fill_diagonal(D_p, np.sqrt(rho))
     
    N_feat = M*N*K # Total number of features
    N_out = K # Number of output labels (users)
    
    # Creating the dataset
    folder = "Dataset_AP_" + str(M) + "_Users_" + str(K)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Filenames for dataset
    test_feat = folder + "/test_feat_MU_" + str(M) + ".pkl"
    test_out = folder + "/test_out_MU_" + str(M) + ".pkl"
    
    # Feature and output labels for DataFrames
    feature = ["feature" + str(i) for i in range(N_feat)]  #for features
    outputs = ["feature" + str(i) for i in range(N_out)]  #for output labels
    
    
    # Generating and saving test samples
    features, labels = generate_samples(testing_samples, L, D, M, K, N, S, D_p, tau, std_shadowing, var_h,var_n,eta)
    test_df = pd.DataFrame(features, columns = feature)
    test_df.to_pickle(test_feat)
    
    test_df = pd.DataFrame(labels, columns = outputs)
    test_df.to_pickle(test_out)
    
if __name__ == "__main__":
    main()
