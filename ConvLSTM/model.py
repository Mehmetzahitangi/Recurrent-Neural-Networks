# -*- coding: utf-8 -*-
"""
@author: mehme
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as func 

def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (
        np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) -1 ) / stride[0] + 1).astype(int),
        np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) -1 ) / stride[1] + 1).astype(int),        
        )
    return outshape

# 2D CNN encoder train from scratch (no transfer learning)
class EncoderCNN(nn.Module):
    
    def __init__(self, img_x=90, img_y = 120, fc_hidden1 = 512, fc_hidden2=512, dropout = 0.3, CNN_embed_dim =300):
        super(EncoderCNN, self).__init__()
        
        self.img_x = img_x
        self.img_y = img_y
        self.CNN_embed_dim = CNN_embed_dim
        
        # CNN Architectures
        self.ch1, self.ch2, self.ch3, self.ch4 = 32,64,128,256
        self.kernel_size1, self.kernel_size2, self.kernel_size3, self.kernel_size4 = (5,5), (3,3), (3,3), (3,3) # 2d kernel size
        self.stride1, self.stride2, self.stride3, self.stride4 = (2,2), (2,2), (2,2), (2,2) # 2d strides
        self.padding1, self.padding2, self.padding3, self.padding4 = (0,0), (0,0), (0,0), (0,0) # 2d padding
        
        # conv2D output shapes
        self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y), self.padding1, self.kernel_size1, self.stride1)
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.padding2, self.kernel_size2, self.stride2)
        self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.padding3, self.kernel_size3, self.stride3)
        self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.padding4, self.kernel_size4, self.stride4)
        
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.dropout = dropout
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = self.ch1, kernel_size = self.kernel_size1, stride = self.stride1, padding=self.padding1),
            nn.BatchNorm2d(self.ch1, momentum = 0.01),
            nn.ReLU(inplace = True),
            #nn.MaxPool2d(kernel_size = 2)
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = self.ch1, out_channels = self.ch2, kernel_size = self.kernel_size2, stride = self.stride2, padding=self.padding2),
            nn.BatchNorm2d(self.ch2, momentum = 0.01),
            nn.ReLU(inplace = True),
            #nn.MaxPool2d(kernel_size = 2)
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = self.ch2, out_channels = self.ch3, kernel_size = self.kernel_size3, stride = self.stride3, padding=self.padding3),
            nn.BatchNorm2d(self.ch3, momentum = 0.01),
            nn.ReLU(inplace = True),
            #nn.MaxPool2d(kernel_size = 2)
            )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = self.ch3, out_channels = self.ch4, kernel_size = self.kernel_size4, stride = self.stride4, padding=self.padding4),
            nn.BatchNorm2d(self.ch4, momentum = 0.01),
            nn.ReLU(inplace = True),
            #nn.MaxPool2d(kernel_size = 2)
            )
        
        self.drop = nn.Dropout2d(self.dropout)
        self.pool = nn.MaxPool2d(2)
        self.fully_connected_1 = nn.Linear(self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1], self.fc_hidden1) # fully connected layer, output k classes
        self.fully_connected_2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fully_connected_3 = nn.Linear(self.fc_hidden2, self.CNN_embed_dim) # output = CNN embedding latent variables
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        
        for i in range(x_3d.size(1)):
            #CNNs
            conv_layer1 = self.conv1(x_3d[:, i, :, :, :])
            conv_layer2 = self.conv2(conv_layer1)
            conv_layer3 = self.conv3(conv_layer2)
            conv_layer4 = self.conv4(conv_layer3)
            conv_layer5 = conv_layer4.view(conv_layer4.size(0), -1) # flatten the output 
            
            #FC Layers
            fc_layer1 = func.relu(self.fully_connected_1(conv_layer5))
            # func.dropout(fc_layer1, p = self.dropout, training = self.training)
            fc_layer2 = func.relu(self.fully_connected_2(fc_layer1))
            fc_layer3 = func.dropout(fc_layer2, p = self.dropout, training = self.training)
            fc_layer4 = self.fully_connected_3(fc_layer3)
            cnn_embed_seq.append(fc_layer4)
            
        # swap time and sample dim such that (sample dim ,time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim =0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)
        
        return cnn_embed_seq
    
class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim = 300, h_RNN_layers = 3, h_RNN = 256, h_FC_dim = 128, dropout=0.3, num_classes = 50):
        super(DecoderRNN, self).__init__()
        
        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers # RNN hidden layers
        self.h_RNN = h_RNN # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.dropout = dropout
        self.num_classes = num_classes
        
        self.LSTM = nn.LSTM(
            input_size = self.RNN_input_size,
            hidden_size = self.h_RNN,
            num_layers = h_RNN_layers,
            batch_first = True # input and output will has batch size as 1s dimension. e.g. (batch,time_step,input_size)
            )
        
        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)
        
    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """
        
        # FC layers
        fully_conn1 = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        fully_conn2 = func.relu(fully_conn1)
        fully_conn3 = func.dropout(fully_conn2, p = self.dropout, training = self.training)
        fully_conn4 = self.fc2(fully_conn3)
        
        return fully_conn4
    
""" DEBUG """

# EncoderCNN architecture parameters
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512  # latent dim extracted by 2D CNN
img_width, img_height = 256, 342  # resize video 2d frame size
dropout_p = 0.0  # dropout probability
# DecoderRNN architecture parameters
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

TOTAL_CLASSES = 3

device = torch.device('cpu') #torch.device("cuda")

cnn_encoder = EncoderCNN(
    img_x=img_width, img_y=img_height, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, dropout=dropout_p, CNN_embed_dim=CNN_embed_dim
).to(device)

rnn_decoder = DecoderRNN(
    CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, h_FC_dim=RNN_FC_dim, dropout=dropout_p, num_classes=TOTAL_CLASSES
).to(device)


x = torch.zeros(size=(1, 28, 3, 256, 342)).to(device)
print("Dummy input shape: ", x.shape)
encoder_output = cnn_encoder(x)
print("Encoder output : ", encoder_output.shape)
output = rnn_decoder(encoder_output)
print("Decoder output : ", output.shape)