import bson.objectid
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

import bson

from custom_packages import ImageProcessor
from services import UserManager

import numpy as np
import io, base64
from PIL import Image as PilImage
import os

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

from keras_facenet import FaceNet


from pymongo import MongoClient

raw_folder_path = os.path.join(os.getcwd(), "res", "raw")
prepped_folder_path = os.path.join(os.getcwd(), "res", "prepped")

def prep():
    """
        returns an array of embeddings and user_ids as a tuple
    """
    
    users_ids = []
    users_face_embedings = []
    
    user_manager = UserManager(mongo_db=mongo_fa_db)
    
    users_ids = user_manager.get_users_id()
    
    #load all images 
    for user_id in users_ids:
        image_filenames =[]
        dir_list = os.listdir(os.path.join(raw_folder_path, user_id))
        
        image_filenames = [f for f in dir_list if f.endswith(".jpeg") or f.endswith(".jpg")]
        
        for im_file in image_filenames:
            
            processor = ImageProcessor(src=os.path.join(raw_folder_path, user_id,im_file), output_folder=os.path.join(prepped_folder_path, user_id), keep_ratio=False)


    #             print(facenet_model.embeddings(processor.get_reshaped_dims()))
    
    
    return []


facenet_model = FaceNet() #facenet512
#setup rootine


"""
create neccesary folders

"""

mongo_client = MongoClient("mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.5")
mongo_fa_db = mongo_client["facialAuth"]

app = Flask(__name__)

user_manager = UserManager(mongo_db=mongo_fa_db)

@app.route('/', methods=['GET'])
def index():
    return "hello world"

CORS(app, support_credentials=True)

@app.route("/signup", methods=["POST"])
@cross_origin(supports_credentials=True)
def signup_handler():
    
    json_data = request.json
    
    new_user_id = str(user_manager.create_user(user_data={"username": json_data.get("username")}))
    
    if not(os.path.exists(os.path.join(raw_folder_path, new_user_id))):
        os.mkdir(os.path.join(raw_folder_path, new_user_id))
        
    for base64Image in json_data.get("images"):
        base64Image = str(base64Image)
        base64Image = base64Image.split("data:image/jpeg;base64,")[-1]
        
        image_id = bson.objectid.ObjectId()
        
        img = PilImage.open(io.BytesIO(base64.decodebytes(bytes(base64Image, "utf-8"))))
        
        img.save(f'{os.path.join(raw_folder_path, new_user_id)}\\{str(image_id)}.jpeg')
    
    #retrain
    
    prep()
    
    return "user created"


@app.route("/login", methods=["GET"])
def login_handler():
    user_manager = UserManager(mongo_db=mongo_fa_db)
    user_found = user_manager.get_user(user_id="6630fd7d1026f3a5a7387450")
    
    return user_found

sample_user_id = "0"

sample_user_raw_folder = os.path.join(os.getcwd(), "res", "raw", f"id_{0}")

sample_user_prepped_folder = os.path.join(os.getcwd(), "res", "prepped", f"id_{0}")

labels = []
face_embeddings = []
