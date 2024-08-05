# Blind User Activity Detection for Grant-Free Random Access in Cell-Free mMIMO Networks
Cell-Free massive MIMO networks have recently emerged as a promising solution to tackle the challenges arising from next-generation massive machine-type communication. A fully grant-free DL-based method for user activity detection in CF-mMIMO networks is proposed. Initially, the known non-orthogonal pilot sequences are used to estimate the channel coefficients between each user and the access points. Then, a deep convolutional neural network (CNN) is used to estimate the activity status of the users. The proposed method is ``blind'', i.e., it is fully data-driven and does not require prior large-scale fading coefficients estimation. 

Full text of the paper can be found on [arXiv]()

# [Deep Learning](https://github.com/wilabcnit/wilab-aud-in-cf-mmimo/blob/main/CNN.py) 
we employ three 2D-Convolution layers consisting of 128, 64, and 32 filters, respectively. We use same padding for the convolutional layers, such that the convolution input and output sizes are equal. We employ a kernel/filter of size 2 x 2 for N > 1 and a filter of size 1 x 2 for N=1, with a stride of 1. After the 2D-convolution layers, we employ a set of linear layer, batch normlization layer, and activation function.

![alt text](https://github.com/wilabcnit/wilab-aud-in-cf-mmimo/blob/main/arch.jpg)
 
# [Dataset Generation](https://github.com/wilabcnit/wilab-aud-in-cf-mmimo/blob/main/dataset_generation.py)
Python script that generates datasets for training, validation, and testing a CNN for active user detection in CF-mMIMO systems.

# How to Run
1. Clone the repository
```
git clone https://github.com/wilabcnit/wilab-aud-in-cf-mmimo
```
2. Navigate to the Test folder
```
cd wilab-aud-in-cf-mmimo
```
4. Execute the [script](https://github.com/wilabcnit/wilab-aud-in-cf-mmimo/blob/main/script)
```
./script
```
The script will generate training, validation, and test datasets. The network will be trained using the training dataset and evaluated using the validation and test datasets. To modify the number of Access Points, users, or antennas at each AP, please adjust the corresponding variables inside the configuration file/script. [script](https://github.com/wilabcnit/wilab-aud-in-cf-mmimo/blob/main/script). 

We tested the network on the server with 
RAM 128 GB
GPU 16 GB memory 
Ubuntu 22.04.1 LTS
python 3.9.16
keras-gpu 2.4.3
tensorflow-gpu 2.4.1

Other dependencies:
scikit-learn
pandas

# Test Example
We have provided a test case with 20 Access Points (APs), 200 users, and 3 antennas per AP. All relevant files are located in the [Test](https://github.com/wilabcnit/wilab-aud-in-cf-mmimo/tree/main/Test) folder. This folder includes a pretrained model configured with the specified system parameters. Please follow the following steps:

1. Clone the repository
```
git clone https://github.com/wilabcnit/wilab-aud-in-cf-mmimo
```
2. Navigate to the Test folder
```
cd wilab-aud-in-cf-mmimo/Test
```
3. Extract the model.zip
```
unzip model.zip
```
4. Execute the [script](https://github.com/wilabcnit/wilab-aud-in-cf-mmimo/blob/main/Test/script)
```
./script
```

We have used pickle v4.0 to create the [pilots20.pkl](https://github.com/wilabcnit/wilab-aud-in-cf-mmimo/blob/main/Test/pilots20.pkl) file. If you encounter any compatibility issues, please ensure that you are using the same version of Pickle.


