import bson.objectid
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

import bson

from custom_packages import ImageProcessor
from services import UserManager, EmbeddingManager

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

def svm_train_classifier():
    user_ids = []
    embeddings = []
    
    user_manager = UserManager(mongo_db=mongo_fa_db)
    emb_manager = EmbeddingManager(mongo_db=mongo_fa_db)
    
    for user_id in user_manager.get_users_id():
        user_emb_dict = emb_manager.read_embedding(user_id=user_id) # user_emb is a dict such that {user_id: str, embedding: [[numbers]]}
        
        fetched_emb_list = np.asarray(user_emb_dict)
        
        print(fetched_emb_list)
        
        # user_ids.extend([user_id] * len(fetched_emb_list))
        # embeddings.append(user_emb_dict)
        
        # le = LabelEncoder().fit(user_ids)
        
        # y = le.transform(user_ids)
        
        # classifier = SVC(kernel='linear', probability=True).fit(embeddings, y)
        
    
    # print(user_ids)
    # print(embeddings)
    
    # return le, classifier



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

    return "user created"


@app.route("/login", methods=["GET"])
def login_handler():
    user_manager = UserManager(mongo_db=mongo_fa_db)
    user_found = user_manager.get_user(user_id="6630fd7d1026f3a5a7387450")
    
    return user_found

# sample_user_id = "0"

# sample_user_raw_folder = os.path.join(os.getcwd(), "res", "raw", f"id_{0}")

# sample_user_prepped_folder = os.path.join(os.getcwd(), "res", "prepped", f"id_{0}")

# labels = []
# face_embeddings = []
