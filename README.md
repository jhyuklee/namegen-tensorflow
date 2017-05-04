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
# Modules

![alt_tag](images/LSTM-h.png)
![alt_tag](images/LSTM-C.png)
- Each name string will be represented into an encoded vector consisted of (h+c)
- h : LSTM's hidden state
- C : LSTM's output state

##1. Encoder

- Encodes the given string value into a hidden vector h
- Output of the model = (2 x cell-dim) = (LSTM's h) + (LSTM's c)

| Division | representation | specifics                                   |
|:--------:|:--------------:|:-------------------------------------------:|
|   input  |       x        | character-level embedding of name strings   |
|  output  |       h        | vector-level representation of name strings |
|   model  |      RNN       | input-size * time-step -> (2 x cell-dim)    |

##2. Decoder

- Decodes the given hidden vector into an approximated string value x-hat
- Output of the model will (time_steps x input_dim)

| Division | representation | specifics                                   |
|:--------:|:--------------:|:-------------------------------------------:|
|  input   |       h        | vector-level representation of name strings |
| output   |     x-hat      | near-value reconstruction of 'x'            |
|  model   |      RNN       | input-dim |

##3. Generator (G)

- Generates a fake hidden vector representing a name string
- Output of the model = (2 x cell-dim) = (LSTM's h) + (LSTM's c)

| Division | representation | specifics                             |
|:--------:|:--------------:|:-------------------------------------:|
|  input   |      Zc        | Random input vector with class info   |
| output   |      Xc        | Generated name hidden vectors         |
|  model   |      Linear    | (z_dim + class_dim) => (cell_dim * 2) |

##4. Discriminator (D)

- Binary classification. Define whether the given input is fake or not.

| Division | representation | specifics                                      |
|:--------:|:--------------:|:----------------------------------------------:|
|  input   |      Xc        | Hidden vector for name-class representation    |
| output   |      Pc        | Probabilities whether the input is fake or not |
|  model   |      Linear    | (cell_dim * 2  + class_dim) => p               |

