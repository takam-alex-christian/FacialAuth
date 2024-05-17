import bson.objectid
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

import bson

from custom_packages import ImageProcessor, cos_similarity
from services import UserManager, EmbeddingManager

import io, base64
from PIL import Image as PilImage
import os

from keras_facenet import FaceNet

from pymongo import MongoClient

import joblib

raw_folder_path = os.path.join(os.getcwd(), "res", "raw")
prepped_folder_path = os.path.join(os.getcwd(), "res", "prepped")


mongo_client = MongoClient(
    "mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.5"
)

mongo_fa_db = mongo_client["facialAuth"]

app = Flask(__name__)

user_manager = UserManager(mongo_db=mongo_fa_db)


trained_le = joblib.load(f"{os.path.join(os.getcwd(), "svm_model")}\\le.gz")
trained_classifier = joblib.load(f"{os.path.join(os.getcwd(), "svm_model")}\\classifier.gz")

CORS(app, support_credentials=True)
@app.route("/", methods=["GET"])
def index():
    return ""


@app.route("/signup", methods=["POST"])
@cross_origin(supports_credentials=True)
def signup_handler():

    json_data = request.json

    new_user_id = str(
        user_manager.create_user(user_data={"username": json_data.get("username")})
    )

    if not (os.path.exists(os.path.join(raw_folder_path, new_user_id))):
        os.mkdir(os.path.join(raw_folder_path, new_user_id))

    for base64Image in json_data.get("images"):
        base64Image = str(base64Image)
        base64Image = base64Image.split("data:image/jpeg;base64,")[-1]

        image_id = bson.objectid.ObjectId()

        img = PilImage.open(io.BytesIO(base64.decodebytes(bytes(base64Image, "utf-8"))))

        img.save(f"{os.path.join(raw_folder_path, new_user_id)}\\{str(image_id)}.jpeg")

    return {"success": True}


@app.route("/login", methods=["POST"])
@cross_origin(supports_credentials=True)
def login_handler():

    facenet_model = FaceNet()

    # first download image
    json_data = request.json

    prediction = None
    
    
    base64Image = str(json_data.get("image"))
    base64Image = base64Image.split("data:image/jpeg;base64,")[-1]

    image_id = bson.objectid.ObjectId()

    img = PilImage.open(io.BytesIO(base64.decodebytes(bytes(base64Image, "utf-8"))))

    img.save(f"{os.path.join(raw_folder_path, "tmp")}\\{str(image_id)}.jpg")

    
    processor = ImageProcessor(src=os.path.join(raw_folder_path, "tmp", f"{str(image_id)}.jpg"))
    
    
    # load label encoder and classifier
    le = joblib.load(f"{os.path.join(os.getcwd(), "svm_model")}\\le.gz")
    classifier = joblib.load(f"{os.path.join(os.getcwd(), "svm_model")}\\classifier.gz")
    
    if processor.found_face_data is not None:
        test_emb = facenet_model.embeddings(processor.get_reshaped_dims())
        
        # print(test_emb)

        prediction = le.inverse_transform(classifier.predict(test_emb))
        
        # compare embeding to any preprocessed image to check for false positive
        
        predicted_user_prepped_folder = os.path.join(os.getcwd(), "res", "prepped", prediction[0])
        predicted_user_raw_folder = os.path.join(os.getcwd(), "res", "raw", prediction[0])
        
        predicted_user_sample_prepped = ImageProcessor(src=f"{predicted_user_prepped_folder}\\{os.listdir(predicted_user_prepped_folder)[0]}", output_folder="")
        
        old_emb = facenet_model.embeddings(predicted_user_sample_prepped.get_reshaped_dims())[0]
        print(len(old_emb))
        print(len(test_emb[0]))
        
        cosine_similarity = cos_similarity(old_emb, test_emb[0])
        
        if cosine_similarity < 0.5:
            verified = True
            img.save(f"{os.path.join(predicted_user_raw_folder, f"{str(image_id)}.jpg")}")
        else:
            verified = False
        
        print(prediction)
        print(cosine_similarity)
        
        
            
        
        return {"authed": True, "user_id": prediction[0], "verified": verified, "username": user_manager.get_user(prediction[0]) }
    # if login is successfull save image as raw
    
    return {"authed": False, "prediction": None}
    


# sample_user_id = "0"

# sample_user_raw_folder = os.path.join(os.getcwd(), "res", "raw", f"id_{0}")

# sample_user_prepped_folder = os.path.join(os.getcwd(), "res", "prepped", f"id_{0}")

# labels = []
# face_embeddings = []
