from flask import Flask

from custom_packages import ImageProcessor

import numpy as np
import os

import cv2

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

from skimage.transform import resize
from keras_facenet import FaceNet




#load pretrained facenet model

facenet_model = FaceNet()

# import 

#setup rootine
"""
create neccesary folders

"""


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "hello world"


sample_user_id = "0"

sample_user_raw_folder = os.path.join(os.getcwd(), "res", "raw", f"id_{0}")

sample_user_prepped_folder = os.path.join(os.getcwd(), "res", "prepped", f"id_{0}")

labels = []
faces = []
#load all images 
for root, dir, files in os.walk(sample_user_raw_folder):
    
    for im_file in files:
        if im_file.endswith(".png") or im_file.endswith(".jpg") or im_file.endswith(".jpeg"):
            processor = ImageProcessor(src=os.path.join(sample_user_raw_folder, im_file), output_folder=sample_user_prepped_folder, keep_ratio=False)
            processor.store_processed_image()
            
