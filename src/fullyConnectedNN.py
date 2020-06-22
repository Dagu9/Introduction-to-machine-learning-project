import torch.nn as nn

class FullyConnectedNN(nn.Module):
    # neural network of fully connected layers
    # layers_size: list of size of each layer e.g. [784, 256, 128, 10]
    def __init__(self, layers_size, num_classes, activation_type=nn.ReLU):
        super(FullyConnectedNN, self).__init__()

        # number of classes
        self.num_classes = num_classes

        activation = activation_type()

        # layers
        layers = []
        for i in range(1,len(layers_size)):
            # fully connected layer
            layers.append(nn.Linear(layers_size[i-1], layers_size[i]))
            # activation function
            layers.append(activation)

        # final layer
        layers.append(nn.Linear(layers_size[-1], num_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        return self.layers(X)
