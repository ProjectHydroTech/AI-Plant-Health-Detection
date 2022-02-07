import os
import glob
import cv2
import argparse
from libraries.inference import detection_model, category_index
import libraries.inference as infer
 
# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding arguments
parser.add_argument("-d", "--dataset", help = "Image Dataset Path")
parser.add_argument("-c", "--cropped", help = "Cropped Image Path")
parser.add_argument("-b", "--boundbox", help = "Image + Bounding Box Path")
parser.add_argument("-s", "--scorethresh", help = "Score Threshold")
parser.add_argument("-nb", "--boundboxnumber", help = "Number of Bounding Boxes")

# Read arguments from command line
args = parser.parse_args()

#Show directories
if args.dataset is None:
    datasetpath = "dataset"
    print("Using Default dataset path : % s" % datasetpath)
else:
    datasetpath = args.dataset
    print("Dataset path is: % s" % datasetpath)
    
if args.cropped is None:
    croppedpath = "cropped"
    print("Using Default cropped path : % s" % croppedpath)
else:
    croppedpath = args.cropped
    print("Cropped path is: % s" % croppedpath)
    
if args.boundbox is None:
    boundboxpath = "boundbox"
    print("Using Default dataset path : % s" % boundboxpath)
else:
    boundboxpath = args.boundbox
    print("Image + Bound Box Path is: % s" % boundboxpath)

if args.scorethresh is None:
    scorethresh = 0.8
    print("Using Default Score Threshold : % s" % scorethresh)
else:
    scorethresh = args.scorethresh
    print("Using Set Score Threshold : % s" % scorethresh)

if args.boundboxnumber is None:
    numberboxes = 20
    print("Using Default Number of Boxes : % s" % numberboxes)
else:
    numberboxes = args.scorethresh
    print("Using Set Score Threshold : % s" % numberboxes)
    
#----------------------------------------------------------#
#----------------Functions---------------------------------#   
    
def getimgpaths(datasetpaths):

    imagepaths = []
    for file in glob.glob(os.path.join(datasetpaths,"*.jpg")) :
        imagepaths.append(file)
    for file in glob.glob(os.path.join(datasetpaths,"*.png")) :
        imagepaths.append(file)
    for file in glob.glob(os.path.join(datasetpaths,"*.jpeg")) :
        imagepaths.append(file)

    print("{} Images Found! {}".format(len(imagepaths), imagepaths))
    return imagepaths

#----------------------------------------------------------#
#----------------Initialising Functions--------------------#

imgpaths = getimgpaths(datasetpath)
num_images = len(imgpaths)

for x in range (0, num_images) :
    image_np = infer.image_load(imgpaths[x])
    detected, image_with_detections = infer.image_detect(image_np, numberboxes, scorethresh, category_index, detection_model)
    infer.image_crop(imgpaths[x], croppedpath, scorethresh, detected)
    
    #remove front path and leave only image string (tail)
    #i.e. cropped\\image.jpg --> image.jpg
    head, sep, tail = imgpaths[x].partition('\\')

    #remove '.jpg' or '.png' and leave only image name (head1)
    #i.e. image.jpg --> image
    head1, sep1, tail1 = tail.partition('.')

    cv2.imwrite(os.path.join(boundboxpath,'{}_boxes.jpg'.format(head1)), image_with_detections)
    print(os.path.join(boundboxpath,'{}_boxes.jpg written!'.format(head1)))
    print("#################################################")
