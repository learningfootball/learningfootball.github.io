

overlays = [ 'distortion.jpg',
             'dust.jpg',
             'noise01.jpg',
             'noise02.jpg',
             'rain01.jpg',
             'rain02.jpg',
             'rain03.jpg',
             'snow01.jpg',
             'snow02.jpg' ]

outputfolder = './examples/'

# each path has 3000 .jpg images 'img1.jpg'... 'img2.jpg', ...
inputpaths = [ '../../trainingdata/01-basic/trainingdata/trainingimages/',
               '../../trainingdata/02-randombackgrounds/trainingdata/trainingimages/',
               '../../trainingdata/03-lighting/trainingdata/trainingimages/' ]

# generate 100 test examples and put in the output folder
numexamples = 100


import cv2
import numpy as np
import random 

for n in range(0, numexamples):

    rpath = inputpaths[ random.randint(0, len(inputpaths)-1 ) ]
    rimg  = 'img' + str(random.randint(1,3000-1)) + '.jpg'
    
    inputimage = rpath + rimg 
    noiseimage = overlays[ random.randint(0, len(overlays)-1 ) ]
    
    # Loading our images
    # Background/Input image
    background = cv2.imread( inputimage ) # e.g. test.jpg
     
    # Overlay image
    overlay_image = cv2.imread( noiseimage ) # e.g., noise01.jpg
     
    # Resize the overlay image to match the bg image dimensions
    overlay_image = cv2.resize(overlay_image, (256, 256))
    h, w = overlay_image.shape[:2]
     
    # Create a new np array
    shapes = np.zeros_like(background, np.uint8)
     
    # Put the overlay at the bottom-right corner
    shapes[background.shape[0]-h:, background.shape[1]-w:] = overlay_image
     
    # Change this into bool to use it as mask
    mask = shapes.astype(bool)
     
     
    alpha = random.uniform(0.1, 0.5) # e.g. 0.2
    # Create a copy of the image to work with
    bg_img = background.copy()
    # Create the overlay
    bg_img[mask] = cv2.addWeighted(bg_img, 
                                   1 - alpha, 
                                   shapes,
                                   alpha, 0)[mask]
                                           
    # resize the image before displaying
    bg_img = cv2.resize(bg_img, (512, 512))
    # cv2.imshow('Final Overlay', bg_img)
    # cv2.waitKey(0)

    filename = outputfolder + 'img' + str(n) + '.jpg'
    cv2.imwrite(filename, bg_img)