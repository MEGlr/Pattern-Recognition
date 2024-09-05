# 2D CNN

# copy paste from convolution.py and add the implementation
import torch
import torch.nn as nn

# definition of Custom CNN
class CNNBackbone(nn.Module):
    # defines model's layers
    def __init__(self, input_dims, in_channels, filters, feature_size):
        
        # use this line so that inheritance works properly
        super(CNNBackbone # model's name
                , self).__init__()

        # saving values
        self.input_dims = input_dims
        self.in_channels = in_channels
        self.filters = filters
        self.feature_size = feature_size
        
        # defining convolutional layers

        # defining 1rst 2D convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, # number of channels of the input image
                     filters[0], # channels produced by the convolution
                     kernel_size=(5,5), stride=1, padding=2),
            nn.BatchNorm2d((self.in_channels**1) * filters[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))


        # defining 2nd 2D convolution layer
        self.conv2 = nn.Sequential(
                nn.Conv2d(filters[0], filters[1], kernel_size=(5,5), stride=1, padding=2),
                nn.BatchNorm2d((self.in_channels**2) * filters[1]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        # defining 3rd 2D convolution layer
        self.conv3 = nn.Sequential(
                nn.Conv2d(filters[1], filters[2], kernel_size=(3,3), stride=1, padding=1),
                nn.BatchNorm2d((self.in_channels**3) * filters[2]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        # defining 4rth 2D convolution layer
        self.conv4 = nn.Sequential(
                nn.Conv2d(filters[2], filters[3], kernel_size=(3,3), stride=1, padding=1),
                nn.BatchNorm2d((self.in_channels**4) * filters[3]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        #print(input_dims, "input dims")
        shape_after_convs = [input_dims[0]//2**(len(filters)), input_dims[1]//2**(len(filters))] 
        #print(shape_after_convs)
        #print("linear", filters[3] * shape_after_convs[0] * shape_after_convs[1])

        # defining fully connected layer
        self.fc1 = nn.Linear(filters[3] * shape_after_convs[0] * shape_after_convs[1], 
                            self.feature_size  # out features : # categories count = # music genres 
                            )
    # Defining the forward pass
    def forward(self, x):
        """
        x : input (4-dimentional [batch_size, channels, height, width])
        defines how the model should compute on input x
        
        """
        
        # dimensions 
        x = x.view(x.shape[0], self.in_channels, x.shape[1], x.shape[2])
        # print(x.shape, "initial shape")
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        #print(out.shape, "-----BEFORE----------------")
        # flatten x 
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        
        # send out model's final predictions
        return out
