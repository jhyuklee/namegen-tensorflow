Name-generation
=====================================================================================================
Repository for Generative model for names of [Olympic Name Dataset]()
Created at March 2, 2017 Korea Unviversity, Data-Mining Lab

Tensorflow Implementation for Name-GAN(Generative Adversarial Network).
Jinhyuk Lee (63coldnoodle@gmail.com)
Bumsoo Kim  (meliketoy@gmail.com)

## Requirements
Take a look at the [installation instruction](./INSTALL.md) for details about installation.
- Install [cuda-8.0](https://developer.nvidia.com/cuda-downlaods)
- Install [cudnn-v5.1](https://developer.nvidia.com/cudnn)
- Install [Tensorflow](https://www.tensorflow.org/install/install_linux)
```bash
# Install Tensorflow GPU version
$ sudo apt-get install python-pip python-dev
$ pip install tensorflow-gpu

# If the code above doesn't work, try
$ sudo -H pip install tensorflow-gpu
$ sudo pip install --upgrade
```
# Download the dataset

# Modules
1. Encoder

| Division | representation | specifics                                   |
|:--------:|:--------------:|:-------------------------------------------:|
|   input  |       x        | character-level embedding of name strings   |
|  output  |       h        | vector-level representation of name strings |
|   model  |      RNN       | input-size * time-step -> (2 x cell-dim)    |
=> Why 2 x cell-dim even though the model is not Bi-LSTM?

2. Decoder

| Division | representation | specifics                                   |
|:--------:|:--------------:|:-------------------------------------------:|
|  input   |       h        | vector-level representation of name strings |
| output   |     x-hat      | near-value reconstruction of 'x'            |
|  model   |      RNN       | input-dim |

3. Generator (G)

| Division | representation | specifics                             |
|:--------:|:--------------:|:-------------------------------------:|
|  input   |      Zc        | Random input vector with class info   |
| output   |      Xc        | Generated name hidden vectors         |
|  model   |      Linear    | (z_dim + class_dim) => (cell_dim * 2) |

4. Discriminator (D)

| Division | representation | specifics                                    |
|:--------:|:--------------:|:--------------------------------------------:|
|  input   |      Xc        | Hidden vector for name-class representation  |
| output   |      Pc        | Probabilities of the                |
|  model   |      Linear    | (cell_dim * 2  + class_dim) => p             |


