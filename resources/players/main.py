
# PyTorches pre-trained models for human pose and keypoint detection
# ref: Keypoint RCNN deep learning model with a ResNet-50 base architecture.

# example:  python main.py --input ./misc/players.jpg
import cv2
import matplotlib


edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6)
]

def draw_keypoints(outputs, image):
    # the `outputs` is list which in-turn contains the dictionaries 
    for i in range(len(outputs[0]['keypoints'])):
        keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()
        # proceed to draw the lines if the confidence score is above 0.9
        if outputs[0]['scores'][i] > 0.9:
            keypoints = keypoints[:, :].reshape(-1, 3)
            for p in range(keypoints.shape[0]):
                # draw the keypoints
                cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 
                            3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                # uncomment the following lines if you want to put keypoint number
                # cv2.putText(image, f"{p}", (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            for ie, e in enumerate(edges):
                # get different colors for the edges
                rgb = matplotlib.colors.hsv_to_rgb([
                    ie/float(len(edges)), 1.0, 1.0
                ])
                rgb = rgb*255
                # join the keypoint pairs to draw the skeletal structure
                #cv2.line(image, (keypoints[e, 0][0], keypoints[e, 1][0]),
                #        (keypoints[e, 0][1], keypoints[e, 1][1]),
                #        tuple(rgb), 2, lineType=cv2.LINE_AA)
                
                c0 = keypoints[e, 0][0]
                c1 = keypoints[e, 1][0]
                c2 = keypoints[e, 0][1]
                c3 = keypoints[e, 1][1]
                # print( c0, c1, c2, c3 )
                c0 = int(c0)
                c1 = int(c1)
                c2 = int(c2)
                c3 = int(c3)
                
                line_thickness = 2
                #cv2.line(image, (c0, c1), (c2, c3), tuple(rgb), thickness=line_thickness)

                cv2.line(image, (c0,c1), (c2,c3), tuple(rgb), 2, lineType=cv2.LINE_AA)
                
        else:
            continue
    return image



import torch
import torchvision
import numpy as np
import cv2
import argparse
from PIL import Image
from torchvision.transforms import transforms as transforms


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, 
                    help='path to the input data')
args = vars(parser.parse_args())


# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# initialize the model
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
                                                               num_keypoints=17)
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modle on to the computation device and set to eval mode
model.to(device).eval()



image_path = args['input']
image = Image.open(image_path).convert('RGB')
# NumPy copy of the image for OpenCV functions
orig_numpy = np.array(image, dtype=np.float32)
# convert the NumPy image to OpenCV BGR format
orig_numpy = cv2.cvtColor(orig_numpy, cv2.COLOR_RGB2BGR) / 255.
# transform the image
image = transform(image)
# add a batch dimension
image = image.unsqueeze(0).to(device)


with torch.no_grad():
    outputs = model(image)
output_image = draw_keypoints(outputs, orig_numpy)
# visualize the image
cv2.imshow('Keypoint image', output_image)
cv2.waitKey(0)



# set the save path
save_path = f"./testresult.jpg"
cv2.imwrite(save_path, output_image*255.)

