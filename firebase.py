import pyrebase
import os
import shutil
from pathlib import Path
import glob
import cv2
import argparse
from libraries.inference import detection_model, category_index
import libraries.inference as infer

#----------------------------------------------------------#
#----------------Firebase auth-----------------------------#   

config = {
  'apiKey': "to be configured",
  'authDomain': "to be configured",
  'databaseURL': "to be configured",
  'projectId': "to be configured",
  'storageBucket': "to be configured",
  'messagingSenderId': "to be configured",
  'appId': "to be configured",
  'measurementId': "to be configured",
  'serviceAccount': "to be configured"
}


firebase = pyrebase.initialize_app(config)
db = firebase.database()
storage = firebase.storage()

auth = firebase.auth()
user = auth.sign_in_with_email_and_password("test@gmail.com", "123456789")

#----------------------------------------------------------#
#----------------Paths/Variables---------------------------#   

datasetpath = "dataset"
cache = ".cache"

scorethresh = 0.8
numberboxes = 20

#----------------------------------------------------------#
#----------------Functions---------------------------------#   
    
def getimgpaths(datasetpaths): #gets all images in directory and concantenates to an array

    imagepaths = []
    for file in glob.glob(os.path.join(datasetpaths,"*.jpg")) :
        imagepaths.append(file)
    for file in glob.glob(os.path.join(datasetpaths,"*.png")) :
        imagepaths.append(file)
    for file in glob.glob(os.path.join(datasetpaths,"*.jpeg")) :
        imagepaths.append(file)

    print("{} Images Found! {}".format(len(imagepaths), imagepaths))
    return imagepaths

def getcachedimages(imagename): #gets all images in directory and concantenates to an array
    
    cachedcroppedpaths = []
    for file in glob.glob(os.path.join(cache,"{}_*_cropped.jpg".format(imagename))) :
        cachedcroppedpaths.append(file)
    num_cropcache = len(cachedcroppedpaths)
    
    return cachedcroppedpaths, num_cropcache

#----------------------------------------------------------#
#----------------Initialising Functions--------------------#

try:
    path = Path(cache) #initialise image .cache for uploading
    path.mkdir(parents=True, exist_ok=False)
except FileExistsError:
    print(".cache already exists! removing now")
    shutil.rmtree(cache) #removes image .cache
    path = Path(cache) #initialise image .cache for uploading
    path.mkdir(parents=True, exist_ok=True)
    pass

imgpaths = getimgpaths(datasetpath)
num_images = len(imgpaths)

for x in range (0, num_images) :
    image_np = infer.image_load(imgpaths[x])
    detected, image_with_detections = infer.image_detect(image_np, numberboxes, scorethresh, category_index, detection_model)
    infer.image_crop(imgpaths[x], cache, scorethresh, detected)
    
    #remove front path and leave only image string (tail)
    #i.e. cropped\\image.jpg --> image.jpg
    head, sep, tail = imgpaths[x].partition('\\')

    #remove '.jpg' or '.png' and leave only image name (head1)
    #i.e. image.jpg --> image
    head1, sep1, tail1 = tail.partition('.')

    cv2.imwrite(os.path.join(cache,'{}_boxes.jpg'.format(head1)), image_with_detections) #write to .cache directory before uploading
    #storage.delete('boundbox/') #clear tempbb directory before writing new image    
    storage.child('boundbox').child('tempbb').put(os.path.join(cache,'{}_boxes.jpg'.format(head1))) #write to tempbb directory
    storage.child('storage').child('imagewithboundbox').child('{}_boxes.jpg'.format(head1)).put(
        os.path.join(cache,'{}_boxes.jpg'.format(head1))) #write to storage directory
    
    print(os.path.join(cache,'{}_boxes.jpg uploaded!'.format(head1)))
    
    cachedcroppedpaths, num_cropcache = getcachedimages(head1)
    for y in range (0, num_cropcache):
                          
        header, seperator, tailname = cachedcroppedpaths[y].partition('\\') 
        #storage.delete('cropped/') #clear tempcrop directory before writing new image
        storage.child('cropped').child('tempcrop{}'.format(y)).put(cachedcroppedpaths[y]) #write to tempcrop directory
        storage.child('storage').child('imagecropped').child(tailname).put(cachedcroppedpaths[y]) #write to storage directory
        print(tailname, " uploaded!")
        
    print("#################################################")

shutil.rmtree(cache) #removes image .cache
