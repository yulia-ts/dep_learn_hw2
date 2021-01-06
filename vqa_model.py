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
# Images need to be shaped into 256x256x3
#this file has 4 classes: Image encoder class, questions encoder class, and final our VQA model class which uses the previous 2, and VGG model class

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

# monochromatic images were resized to RGB and have  in_channels = 3

class MY_VGG(nn.Module):
    def __init__(self, vgg_type= VGG_types["VGG16"], vgg_out=1024):
        super(MY_VGG, self).__init__()
        self.layers = []
        self.in_channels = 3
        # Define the fully connected part of VGG
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, vgg_out),
        )
        ##VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # max pooling (kernel_size, stride)
        self.pool = nn.MaxPool2d(2, 2)

        """for x in vgg_type:  # Architecture is my chosen net list
            if type(x) == int:
                out_channels = x
                # Batchnorm and relu are not a part of the original VGG
                self.layers += [nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1), ), nn.BatchNorm2d(x), nn.ReLU(), ]
                in_channels = x
            elif x == "M":
                self.layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]"""

    def forward(self, x, training=True):
        vgg_layers = nn.Sequential(*self.layers)
        #x = vgg_layers(x).to('cuda')
        ##
        #print(x.size())
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        #print(x.size())
        x = self.pool(x)
        #print("first pool")
        #print(x.size())
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool(x)
        #print(x.size())
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool(x)
        #print(x.size())
        x = x.view(-1, 8 * 8 * 512)
        x = self.classifier(x)
        return x

class VQANET_Encoder(nn.Module):
    def __init__(self):
        """Images encoder
        """
        super(VQANET_Encoder, self).__init__()
        self.vgg_model = MY_VGG(VGG_types["VGG16"])

    def forward(self, image):
        """Get image feature vector .
        """
        with torch.no_grad():
            img_feature = self.vgg_model(image).to('cuda')  # [batch_size, embed_size]

        #l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True)
        img_feature = img_feature.div(l2_norm)  # l2-normalized feature vector

        return img_feature

class QEncoder(nn.Module):
##question encoder LSTM net
    def __init__(self, q_voc_size, word_emb_size, emb_size, num_layers, hidden_size):

        super(QEncoder, self).__init__()
        ##to check if it can be used
        self.wordtovec = nn.Embedding(q_voc_size, word_emb_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_emb_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, emb_size)     # 2 for hidden and cell states

    def forward(self, question):

        question_v = self.wordtovec(question)                             # [batch_size, max_qst_length=26, word_embed_size=300]
        question_v = self.tanh(question_v)
        question_v = question_v.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        _, (hidden, cell) = self.lstm(question_v)                        # [num_layers=2, batch_size, hidden_size=512]
        question_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        question_feature = question_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        question_feature = question_feature.reshape(question_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        question_feature = self.tanh(question_feature)
        question_feature = self.fc(question_feature)                            # [batch_size, embed_size]

        return question_feature

class VQA_NET(nn.Module):

    def __init__(self, embed_size, q_voc_size, ans_voc_size, word_emb_size, num_layers, hidden_size):

        super(VQA_NET, self).__init__()
        self.image_encoder = VQANET_Encoder()
        self.question_encoder = QEncoder(q_voc_size, word_emb_size, embed_size, num_layers, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_voc_size)
        self.fc2 = nn.Linear(ans_voc_size, ans_voc_size)

    def forward(self, img, question):
        #print("img size")
        #print(img.size())
        image_f = self.image_encoder(img)                     # [batch_size, embed_size]
        question_f = self.question_encoder(question)                     # [batch_size, embed_size]
        comb_feature = torch.mul(image_f, question_f)  # [batch_size, embed_size]
        comb_feature = self.tanh(comb_feature)
        comb_feature = self.dropout(comb_feature)
        comb_feature = self.fc1(comb_feature)           # [batch_size, ans_vocab_size]
        comb_feature = self.tanh(comb_feature)
        comb_feature = self.dropout(comb_feature)
        comb_feature = self.fc2(comb_feature)           # [batch_size, ans_vocab_size]

        return comb_feature
