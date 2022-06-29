# import the required libraries
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # pip install matplotlib
import matplotlib.image as mpimg

import cv2 # pip install opencv-python


print('Football Keypoints');

# --------------------


key_pts_frame = pd.read_csv('trainingdata/imagepoints.csv')

n = 0
image_name = key_pts_frame.iloc[n, 0]
key_pts = key_pts_frame.iloc[n, 1:].values
key_pts = key_pts.astype('float').reshape(-1, 2)

print('Image name: ', image_name)
print('Landmarks shape: ', key_pts.shape)
print('First 4 key pts: {}'.format(key_pts[:4]))

# -------------------------


# print out some stats about the data
print('Number of images: ', key_pts_frame.shape[0])

# -------

def show_keypoints(image, key_pts):
    """Show image with keypoints"""
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')
    
# Display a few different types of images by changing the index n

# select an image by index in our data frame
n = 0
image_name = key_pts_frame.iloc[n, 0]
key_pts = key_pts_frame.iloc[n, 1:].values
key_pts = key_pts.astype('float').reshape(-1, 2)

plt.figure(figsize=(5, 5))
show_keypoints(mpimg.imread(os.path.join('trainingdata/trainingimages/', image_name)), key_pts)
# plt.show()



# ------------

from torch.utils.data import Dataset, DataLoader # pip install torch

class FootballKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample

# ----------

# Construct the dataset
football_dataset = FootballKeypointsDataset(csv_file='trainingdata/imagepoints.csv',
                                        root_dir='trainingdata/trainingimages/')

# print some stats about the dataset
print('Length of dataset: ', len(football_dataset))


# Display a few of the images from the dataset
num_to_display = 3

for i in range(num_to_display):
    
    # define the size of images
    fig = plt.figure(figsize=(20,10))
    
    # randomly select a sample
    rand_i = np.random.randint(0, len(football_dataset))
    sample = football_dataset[rand_i]

    # print the shape of the image and keypoints
    print(i, sample['image'].shape, sample['keypoints'].shape)

    ax = plt.subplot(1, num_to_display, i + 1)
    ax.set_title('Sample #{}'.format(i))
    
    # Using the same display function, defined earlier
    show_keypoints(sample['image'], sample['keypoints'])
    
# ---------------------------


   
import torch
from torchvision import transforms, utils # pip install torchvision
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0


        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}
                

# -----------------------------------

# test out some of these transforms
rescale = Rescale(100)
crop = RandomCrop(50)
composed = transforms.Compose([Rescale(250),
                               RandomCrop(224)])

# apply the transforms to a sample image
test_num = 500
sample = football_dataset[test_num]

fig = plt.figure()
for i, tx in enumerate([rescale, crop, composed]):
    transformed_sample = tx(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tx).__name__)
    show_keypoints(transformed_sample['image'], transformed_sample['keypoints'])

# plt.show()

# -------------------------

# define the data tranform
# order matters! i.e. rescaling should come before a smaller crop
data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])
                     
# create the transformed dataset
transformed_dataset = FootballKeypointsDataset(csv_file='trainingdata/imagepoints.csv',
                                             root_dir='trainingdata/trainingimages/',
                                             transform=data_transform)
                                             

# --------------------

# print some stats about the transformed data
print('Number of images: ', len(transformed_dataset))

# make sure the sample tensors are the expected size
for i in range(5):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())

# --------------

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


## TODO: define the data_transform using transforms.Compose([all tx's, . , .])
# order matters! i.e. rescaling should come before a smaller crop
data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])

# testing that you've defined a transform
assert(data_transform is not None), 'Define a data_transform'


# ---------------------------

# create the transformed dataset
transformed_dataset = FootballKeypointsDataset(csv_file='trainingdata/imagepoints.csv',
                                             root_dir='trainingdata/trainingimages/',
                                             transform=data_transform)


print('Number of images: ', len(transformed_dataset))

# iterate through the transformed dataset and print some stats about the first few samples
for i in range(4):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())


# -------


# load training data in batches
batch_size = 50

train_loader = DataLoader(transformed_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0)



# -------------------------------




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
     

import torch
import torch.nn as nn
import torch.nn.functional as F


    
## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you
# from models import Net

net = Net()
print(net)

# -----------------------


#from torchsummary import summary

#summary(net, (1, 250, 224))
    
# -----------------------


# test the model on a batch of test images

def net_sample_output():
    
    # iterate through the test dataset
    for i, sample in enumerate(train_loader):
        
        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']

        # convert images to FloatTensors
        images = images.type(torch.FloatTensor)

        # forward pass to get net output
        output_pts = net(images)
        
        # reshape to batch_size x 26 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 26, -1)
        
        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts


# call the above function
# returns: test images, test predicted keypoints, test ground truth keypoints
test_images, test_outputs, gt_pts = net_sample_output()

# print out the dimensions of the data to see if they make sense
print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())


# -----------------


def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')
        
        
# visualize the output
# by default this shows a batch of 10 images
def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):

    for i in range(batch_size):
        plt.figure(figsize=(20,10))
        ax = plt.subplot(1, batch_size, i+1)

        # un-transform the image data
        image = test_images[i].data   # get the image from it's wrapper
        image = image.numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints  
        predicted_key_pts = predicted_key_pts*50.0+100
        
        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]         
            ground_truth_pts = ground_truth_pts*50.0+100
        
        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)
            
        plt.axis('off')

    plt.show()
    
# call it
# visualize_output(test_images, test_outputs, gt_pts)      
        
       
# ---------------



## TODO: Define the loss and optimization
import torch.optim as optim

criterion = nn.SmoothL1Loss()

optimizer = optim.Adam(net.parameters())

# --------------------

# at beginning of the script
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import os 
os.environ["CUDA_VISIBLE_DEVICES"]=""

device = torch.device("cpu")


import time

def train_net(n_epochs):

    device = torch.device("cpu")
    
    # prepare the net for training
    net.to(device)
    net.train()
    
    start_time = time.time()
    loss_list = []
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image'].to(device)
            key_pts = data['keypoints'].to(device)

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

# Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
# unsolved
    
            # convert variables to floats for regression loss
            #key_pts = key_pts.type(torch.cuda.FloatTensor)
            #images = images.type(torch.cuda.FloatTensor)
            
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            # to convert loss into a scalar and add it to the running_loss, use .item()
            running_loss += loss.item()
            
            print_every = 10
            if batch_i % print_every == 9:    # print every 10 batches
                avg_running_lost = running_loss/print_every
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, avg_running_lost))
                loss_list.append(avg_running_lost)
                running_loss = 0.0

    print('Finished Training in {} seconds'.format(time.time() - start_time))
    
    plt.figure(figsize=(20,10))
    plt.plot(loss_list)


# -------------

# train your network
n_epochs = 100 # 500 # start small, and increase when you've decided on your model structure and hyperparams

# if we want to train the data
if 0:
    train_net(n_epochs)

## hack ============

if 1:
    ## TODO: change the name to something uniqe for each new model
    model_dir = 'savedmodels/'
    model_name = 'keypoints_model_adam_SmoothL1_50_batchsize_100_epochs_12800_6400_1000_dense.pt'

    ## TODO: load the best saved model parameters (by your path name)
    ## You'll need to un-comment the line below and add the correct name for *your* saved model
    net.load_state_dict(torch.load(model_dir+model_name))

    ## print out your net and prepare it for testing (uncomment the line below)
    net.eval()



# ----------------------
 
 
 # get a sample of test data again
net.cpu()
test_images, test_outputs, gt_pts = net_sample_output()

print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())



# ---------------

## TODO: visualize your test output
# you can use the same function as before, by un-commenting the line below:

visualize_output(test_images, test_outputs, gt_pts)



# -----------------

if 0:
    ## TODO: change the name to something uniqe for each new model
    model_dir = 'savedmodels/'
    model_name = 'keypoints_model_adam_SmoothL1_50_batchsize_100_epochs_12800_6400_1000_dense.pt'

    # after training, save your model parameters in the dir 'saved_models'
    torch.save(net.state_dict(), model_dir+model_name)




    