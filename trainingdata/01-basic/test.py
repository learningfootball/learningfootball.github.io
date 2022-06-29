

#import matplotlib.pyplot as plt

#data = [[0, 0.25], [0.5, 0.75]]

#fig, ax = plt.subplots()
#im = ax.imshow(data, cmap=plt.get_cmap('hot'), interpolation='nearest',
#               vmin=0, vmax=1)
#fig.colorbar(im)
# plt.show()

# --------------


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
# load in color image for face detection
# image = cv2.imread('misc/test01.jpg')
image = cv2.imread('misc/stadium.jpg')


# switch red and blue color channels 
# --> by default OpenCV assumes BLUE comes first, not RED as in many images
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plot the image
#fig = plt.figure(figsize=(9,9))
#plt.imshow(image)
# plt.show()




# -------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 52 values, 2 for each of the 26 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # with stride of 1, shape becomes (224 - 5) / 1 + 1 = 220
        self.conv1 = nn.Conv2d(1, 32, 5)
        # shape becomes 220/2 = 110
        self.maxpool = nn.MaxPool2d(2, 2)
        # (110 - 3) / 1 + 1 = 108
        # shape becomes 108/2 = 54
        self.conv2 = nn.Conv2d(32, 64, 3)
        # (54 - 3) / 1 + 1 = 52
        # shape becomes 52/2 = 26
        self.conv3 = nn.Conv2d(64, 128, 3)
        # (26 - 3) / 1 + 1 = 24
        # shape becomes 24/2 = 12
        self.conv4 = nn.Conv2d(128, 256, 3)
        # (12 - 3) / 1 + 1 = 10
        # shape becomes 10/2 = 5
        self.conv5 = nn.Conv2d(256, 512, 3)
        # 5*5*512 = 12800
        self.fc1 = nn.Linear(12800, 6400)
        self.fc2 = nn.Linear(6400, 1000)
        self.output = nn.Linear(1000, 52)

        self.dropout1_2 = nn.Dropout(0.1)
        self.dropout3_4_5 = nn.Dropout(0.3)
        self.dropout6_7 = nn.Dropout(0.5)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting


    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.dropout1_2(x)
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.dropout1_2(x)
        x = self.maxpool(F.relu(self.conv3(x)))
        x = self.dropout3_4_5(x)
        x = self.maxpool(F.relu(self.conv4(x)))
        x = self.dropout3_4_5(x)
        x = self.maxpool(F.relu(self.conv5(x)))
        x = self.dropout3_4_5(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout6_7(x)
        x = F.relu(self.fc2(x))
        x = self.dropout6_7(x)
        x = self.output(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
     
     
# Loading in a trained model
# import torch
# from models import Net

net = Net()
# print(net)

## TODO: load the best saved model parameters (by your path name)
## You'll need to un-comment the line below and add the correct name for *your* saved model
net.load_state_dict(torch.load('savedmodels/keypoints_model_adam_SmoothL1_50_batchsize_100_epochs_12800_6400_1000_dense.pt'))

## print out your net and prepare it for testing (uncomment the line below)
net.eval()

# -----------------------



# -----------------------

def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=50, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')
        
        
# --------------------------


# Select the region of interest that is the face in the image 
# roi = image_copy[y:y+h, x:x+w]
roi = image

## TODO: Convert the face region from RGB to grayscale
roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
padding = 20
roi = cv2.copyMakeBorder(roi, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
## TODO: Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
roi = roi/255.0
## TODO: Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
roi = cv2.resize(roi, (224, 224))
## TODO: Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
roi_copy = roi.copy()
if(len(roi.shape) == 2):
    roi = roi.reshape(1, roi.shape[0], roi.shape[1], 1)
roi = roi.transpose((0, 3, 1, 2))
## TODO: Make facial keypoint predictions using your loaded, trained network 
## perform a forward pass to get the predicted facial keypoints
roi = torch.from_numpy(roi)
roi = roi.type(torch.FloatTensor)
predicted_key_pts = net(roi)
predicted_key_pts = predicted_key_pts.view(predicted_key_pts.size()[0], 26, -1)
predicted_key_pts = predicted_key_pts.data
predicted_key_pts = predicted_key_pts.numpy()
predicted_key_pts = predicted_key_pts*50.0+100
## TODO: Display each detected face and the corresponding keypoints        
plt.figure(figsize=(20,10))
# ax = plt.subplot(1, len(faces), i+1)
# plt.imshow(image)

# ax.axis('off')
show_all_keypoints(roi_copy, np.squeeze(predicted_key_pts), gt_pts=None)

plt.show()