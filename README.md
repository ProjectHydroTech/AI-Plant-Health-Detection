# AI-Plant-Health-Detection

This python script performs inference and crops and draws bounding boxes on images you provide. 

In this project the script is used for leaf detection and cropping, However this can also be used for any model u have trained. 

## How to use

Clone this repository and copy ur trained model ckpt, ckpt index, pipeline config and labelmap.pbtxt into the libraries/model folder. 

It should look like ckpt-0.data-00000-of-00001, ckpt-0.index and pipeline.config respectively, otherwise rename them.

Run to see the available arguements.
```
$python run.py -h 
```

Otherwise running this will load the default setup with the default folders at /dataset, /cropped and /boundbox
```
$python run.py
```


