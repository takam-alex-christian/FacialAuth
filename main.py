from flask import Flask

#
from custom_packages import ImageProcessor
from services import UserManager

import numpy as np
import os

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

from keras_facenet import FaceNet


from pymongo import MongoClient



facenet_model = FaceNet() #facenet512
#setup rootine


"""
create neccesary folders

"""

mongo_client = MongoClient("mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.5")
mongo_fa_db = mongo_client["facialAuth"]

user_manager = UserManager(mongo_db=mongo_fa_db)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "hello world"

@app.route("/signup", methods=["POST"])
def signup_handler():
    sample_user = user_manager.create_user(user_data={"username": "Eric"})
    print(sample_user)
    return "user created"

@app.route("/login", methods=["GET"])
def login_handler():
    user_found = user_manager.get_user(user_id="6630fd7d1026f3a5a7387450")
    
    return user_found

sample_user_id = "0"

sample_user_raw_folder = os.path.join(os.getcwd(), "res", "raw", f"id_{0}")

sample_user_prepped_folder = os.path.join(os.getcwd(), "res", "prepped", f"id_{0}")

labels = []
face_embeddings = []


def prep():
    """
        returns an array of embeddings and user_ids as a tuple
    """
    
    user_manager = UserManager(mongo_db=mongo_fa_db)
    
    users_ids = user_manager.get_users_id()
    
    #load all images 
    for user_id in users_ids:
        print(user_id)
        
    # for root, dir, files in os.walk(sample_user_raw_folder):
        
    #     for im_file in files:
    #         if im_file.endswith(".png") or im_file.endswith(".jpg") or im_file.endswith(".jpeg"):
    #             processor = ImageProcessor(src=os.path.join(sample_user_raw_folder, im_file), output_folder=sample_user_prepped_folder, keep_ratio=False)
    #             processor.store_processed_image()

    #             print(facenet_model.embeddings(processor.get_reshaped_dims()))
    
    
    return []


prep()