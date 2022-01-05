import os
import tensorflow as tf
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import numpy as np
import matplotlib.pyplot as plt

print('inference.py successfully imported')

#---------------------------------------------------------#
#---------------Variables/Paths---------------------------#

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file("libraries/model/pipeline.config")
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

#load categories
category_index = label_map_util.create_category_index_from_labelmap("libraries/model/label_map.pbtxt")

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore('libraries/model/ckpt-0').expect_partial()
    
#---------------------------------------------------------#
#---------------Functions---------------------------------#

@tf.function
def detect_fn(image):
    #print("detect_fn function loaded")
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections 

def image_load(img_path):
  img = cv2.imread(img_path) #reads image
  img_np = np.array(img) #converts image to array
  #print("image_load function loaded")
  return img_np
  
def image_detect(imagenp, numboxes, scorethresh, cat_index, detection_model):
  #print("image_detect function loaded")
  input_tensor = tf.convert_to_tensor(np.expand_dims(imagenp, 0), dtype=tf.float32) #convert image array to tensor
  detections = detect_fn(input_tensor) #send tensor to detect function above
  
  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
  detections['num_detections'] = num_detections

  # detection_classes should be ints.
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

  label_id_offset = 1
  image_np_with_detections = imagenp.copy()

  viz_utils.visualize_boxes_and_labels_on_image_array(
              image_np_with_detections,
              detections['detection_boxes'],
              detections['detection_classes']+label_id_offset,
              detections['detection_scores'],
              cat_index,
              use_normalized_coordinates=True,
              max_boxes_to_draw=numboxes,
              min_score_thresh=scorethresh,
              agnostic_mode=False)
  
  #print("image_detect function ended")
  return detections, image_np_with_detections

def image_crop(IMAGE_PATH, CROP_PATH, scorethreshold, detection) :
  #print("image_crop loaded")
  image = cv2.imread(IMAGE_PATH) #reads image

  #gets dimensions of image
  height = image.shape[0]
  width = image.shape[1]
  
  #remove front path and leave only image string (tail)
  #i.e. cropped\\image.jpg --> image.jpg
  head, sep, tail = IMAGE_PATH.partition('\\')
  print("Cropping for {}".format(tail))

  #remove '.jpg' or '.png' and leave only image name (head1)
  #i.e. image.jpg --> image
  head1, sep1, tail1 = tail.partition('.')

  for count in range (0, 100) :
    if (detection['detection_scores'][count]) >= scorethreshold :
      #gets normalised boxes and scaled according to image dimensions
      top_y = int((detection['detection_boxes'][count][0])*height)
      top_x = int((detection['detection_boxes'][count][1])*width)
      bot_y = int((detection['detection_boxes'][count][2])*height)
      bot_x = int((detection['detection_boxes'][count][3])*width)
      imagecrop = image[top_y:bot_y, top_x:bot_x]
        
      detectscore = round((detection['detection_scores'][count]), 4)
    
      cv2.imwrite(os.path.join(CROP_PATH,"{}_{}_cropped.jpg".format(head1, detectscore)), imagecrop)
      print(os.path.join(CROP_PATH,"{}_{}_cropped.jpg written!".format(head1, detectscore)))
