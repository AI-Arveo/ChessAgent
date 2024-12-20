from torch import nn
import torch

class NeuralNetwork(nn.Module):
    # num_moves zijn het aantal legale moves die op die beurt beschikbaar zijn
    def __init__(self, input_channels, board_size, num_moves):
        super().__init__()
        # we kiezen voor convolution layers, omdat het neuraal netwerk dan beter de spatial relationships/patterns
        # tussen pieces gaat leren

        # gebruik sequential om meerdere layers na elkaar te gebruiken
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1), # 1ste convolutionl layer
                                    # ReLU introduceert niet-lineariteit, waardoor het netwerk complexe functies kan leren
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # 2de convolutional layer
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
                                    # 3de convolutional layer
                                    nn.ReLU(),
                                    # maxPool2d om dimensies te verkleinen
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Flatten(),
                                    nn.Linear(in_features=board_size * board_size * 64,out_features= 128), # 1ste fully connected layer
                                    # er is geen activation function nodig bij lineare/fully connected lagen
                                    nn.Linear(in_features=128, out_features=512), # 2ste fully connected layer
                                    nn.Linear(in_features=512, out_features=num_moves)) # 3ste fully connected layer

    def forward(self, x):
        self.layer1(x)
        self.layer2(x)
        return nn.Softmax(dim=1)(x)

    # Wordt gebruikt om de parameters van ons netwerk te optimaliseren (denk aan het aantal layers, nodes, type of layers?)
    def optimize(self):
        pass

    # kan mogelijks gebruikt worden om ons model op te slagen. Maar moet hier nog wat meer onderzoek naar doen
    def save(self, path):
        torch.save(self.state_dict(), path)
