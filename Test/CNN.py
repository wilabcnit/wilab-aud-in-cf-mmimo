"""
%% ----------- CNN Training and Testing ---------------
% Written by Muhammad Usman Khan
% Email id: muhammadusman.khan8@unibo.it
% Supervisor: Prof. Marco Chiani
% Department of Electrical, Electronic, and Information Engineering
% "Guglielmo Marconi" - DEI
% University of Bologna, Cesena Campus, Italy
% Version 1.0
% Created On: Feb 19, 2024
% Modified On:
% Description: CNN for active user detection in CF-mMIMO.
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from shutil import copyfile
from numpy.random import seed
import matplotlib.pyplot as plt
import os
import sys
import time
from sklearn.metrics import classification_report, confusion_matrix

#%% Parameters
D = 1000 #area under consideration
M = int(sys.argv[1]); #No. of APs in Cell-Free
k = int(sys.argv[2]); # No. of users
N = int(sys.argv[3]); #No. of antennas at the AP
rho = 23 #Transmit Power of the UE in dBm 
tau = 40; #length of the preamle sequence
batch = 256  #batch size
epochs = 10 #No. of epochs
alpha = 500 # Specify the number of neurons in the network
training_samples = 3*10**6 # No. of training samples
testing_samples = 10**4 # No. of testing samples
valid_samples = 10**4 # No. of validation set 
precision= 3 # Precision for dispalying numbers

#%% Variables
N_feat = M*3*k
N_out = k
rho_lin = 10**((rho-30)/10) # Transmit power in linear scale

# Folder and file paths for dataset
folder = "Dataset_AP_" + str(M) + "_Users_" + str(k)
test_feat = folder + "/test_feat_MU_" + str(M) + ".pkl"
test_out = folder + "/test_out_MU_" + str(M) + ".pkl"
valid_feat = folder + "/valid_feat_MU_" + str(M) + ".pkl"
valid_out = folder + "/valid_out_MU_" + str(M) + ".pkl"

# Feature and output labels
feature = ["feature" + str(i) for i in range(N_feat)]  #for features
outputs = ["feature" + str(i) for i in range(N_out)]  #for output labels
pilots = ["feature" + str(i) for i in range(k)]  #for output labels

files = int(training_samples/batch)

# For reproducible result
seed(2)
tf.random.set_seed(1)


#%% Functions
def roc_curve(labels, y_pred):
    """
    Compute ROC curve.
    """
    thresholds = np.arange(0.0, 1.0, 0.001)
    recall = np.zeros(len(thresholds))
    false_alarm = np.zeros(len(thresholds))
    inc = 0
    for th in thresholds:
        C_temp = np.zeros_like(y_pred)
        C_temp[y_pred >= th] = 1
    
        tp = np.sum((C_temp == 1) &  (labels == 1))
        tn = np.sum((C_temp == 0) &  (labels == 0))
        fp = np.sum((C_temp == 1) &  (labels == 0))
        fn = np.sum((C_temp == 0) &  (labels == 1))
    
        recall[inc] = tp / (tp + fn)
        false_alarm[inc] = fp / (fp + tn)
        inc = inc +1
    return false_alarm, recall


def print_metrics(y_true, y_pred):
    """
    Print evaluation metrics for model predictions.
    """
    false_alarm1, recall1 = roc_curve(y_true, y_pred)
    y_pred = (y_pred>0.5).astype("float")      
    a1 = np.reshape(y_pred,-1)
    b1 = np.reshape(y_true, -1)
    conf1 = confusion_matrix(b1, a1)/(np.shape(a1)[0]); 
    r1 = conf1[1][1]/(conf1[1][0] + conf1[1][1]);
    f1 = conf1[0][1]/(conf1[0][0] + conf1[0][1]);
    print ("Recall",  r1 )
    print ("False Alarm", f1  )
    print("Confusion Matrix:")
    print(conf1)
    print("Classification Report")
    print(classification_report( b1, a1))
    
    
    # Saving Variables
    np.savetxt(direc+"/recall.csv",recall1 ,delimiter=",")
    np.savetxt(direc+"/false_alarm.csv",false_alarm1 ,delimiter=",")
    
    
    # Plotting
    plt.plot(false_alarm1, recall1, label='CNN')
    plt.xlabel('False Alarm')
    plt.ylabel('Recall')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(direc+"/plot.pdf")
  
    return
    
          
    
#%% Main Functions
# Extract Current File name
file = os.path.basename(__file__)
file = os.path.splitext(file)[0]

# Creating folder name same as file name 
direc = './' + file + "_AP_" + str(M) + "_Users_" + str(k) + "_Ant_" + str(N)

# Creating director with same name as file for storing weights and logs
if not os.path.exists(direc):
    os.makedirs(direc)

# Copy the code file to the directory
copyfile(file+'.py', direc + '/' + file +'.py')

# Redirect the output to a file
orig_stdout = sys.stdout
f = open(direc +'/output.out', 'w')
sys.stdout = f



#%% Testing the model
# Load the saved model
loaded_model = keras.models.load_model('model.h5' , compile = False)

# Load Testing dataset
data = pd.read_pickle(test_feat) 
X_test = data[feature]
X_test = X_test.to_numpy()
X_test = X_test.reshape((-1,M,3,k), order = 'C')/np.sqrt(rho_lin)
if (N < 3):
    X_test = X_test[:,:,0:N,:]

data = pd.read_pickle(test_out)
Y_test = data[outputs]
Y_test = Y_test.to_numpy()


# Measure the time taken to make predictions
start_time = time.time()

# Predict the outputs for the test set
predicted = loaded_model.predict(X_test)

#Time out
print("--- Test Time: %s seconds ---" % (time.time() - start_time))
print("--- Test Time per sample: %s seconds ---" % ((time.time() - start_time)/ np.shape(X_test)[0]))

# Print metrics for the test set
print_metrics(Y_test, predicted)

# Redirect to the correct standard output
sys.stdout = orig_stdout
f.close()
