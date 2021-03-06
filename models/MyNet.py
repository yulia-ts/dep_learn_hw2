# Imports
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Processing part- A VGG implementation with dropouts, batchnorm, and relu activation
# Images needs to be shaped into 224x224x3
VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

# if image is monochromatic change in_channels to 1, vgg_out is adjusted!!!! don't forget


class VQANET(nn.Module):
    def __init__(self, image_in_channels=3, vgg_out=1024, text_in_channels=1, terminal_in_channels=2, terminalnet_out=1000):
        super(VQANET, self).__init__()
        self.image_in_channels = image_in_channels
        self.terminal_in_channels = terminal_in_channels        # Define the convolution layers
        self.conv_layers = self.create_conv_layers(VGG_types["VGG16"])  # choose your specific net here
        # Define the fully connected part of VGG
        self.fcsvgg = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, vgg_out),
        )
        # Language Processing part
        # Information can be found at- https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        EncoderRNN = nn.LSTM(*args, **kwargs)
        DecoderRNN = nn.LSTM(*args, **kwargs)

        # Attention decoder
        class AttnDecoderRNN(nn.Module):
            def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
                super(AttnDecoderRNN, self).__init__()
                self.hidden_size = hidden_size
                self.output_size = output_size
                self.dropout_p = dropout_p
                self.max_length = max_length

                self.embedding = nn.Embedding(self.output_size, self.hidden_size)
                self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
                self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
                self.dropout = nn.Dropout(self.dropout_p)
                self.gru = nn.GRU(self.hidden_size, self.hidden_size)
                self.out = nn.Linear(self.hidden_size, self.output_size)

            def forward(self, input, hidden, encoder_outputs):
                embedded = self.embedding(input).view(1, 1, -1)
                embedded = self.dropout(embedded)

                attn_weights = F.softmax(
                    self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
                attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                         encoder_outputs.unsqueeze(0))

                output = torch.cat((embedded[0], attn_applied[0]), 1)
                output = self.attn_combine(output).unsqueeze(0)

                output = F.relu(output)
                output, hidden = self.gru(output, hidden)

                output = F.log_softmax(self.out(output[0]), dim=1)
                return output, hidden, attn_weights

            def initHidden(self):
                return torch.zeros(1, 1, self.hidden_size, device=device)

        #Define the terminal fully connected part
        self.fcterminal = nn.Sequential(
            nn.Linear(1024, terminalnet_out),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(terminalnet_out, terminalnet_out),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(terminalnet_out, terminalnet_out),
            nn.Softmax(dim=terminalnet_out)
        )

    def forwardvgg(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcsvgg(x)
        return x

    def forwardterminal(self, x):
        x = x[:, 0]*x[:, 1]  # This should be an elementwise multiplication
        x = self.fcterminal(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:  # Architecture is my chosen net list
            if type(x) == int:
                out_channels = x
                # Batchnorm and relu are not a part of the original VGG
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),), nn.BatchNorm2d(x), nn.ReLU(), ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)
