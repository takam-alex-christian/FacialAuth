
import numpy as np
import os

from keras_facenet import FaceNet

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, Normalizer

from custom_packages import ImageProcessor
from services import UserManager

import joblib


raw_folder_path = os.path.join(os.getcwd(), "res", "raw")
prepped_folder_path = os.path.join(os.getcwd(), "res", "prepped")

test_folder_path = os.path.join(os.getcwd(), "test", "test_in")
facenet_model = FaceNet()

from pymongo import MongoClient

mongo_client = MongoClient("mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.5")
mongo_fa_db = mongo_client["facialAuth"]

def prepz_test_frame():
    """takes in a filepath and returns embeddings"""
    
    processor = ImageProcessor(src=os.path.join(test_folder_path, "test.jpg"), output_folder=os.path.join(os.getcwd(), "test", "test_out"))
    
    if processor.found_face_data is not None:
       return facenet_model.embeddings(processor.get_reshaped_dims())
    else: 
        return None
    
def trainer():
    """
    :return: list of embeddings
    """
    
    labels = []
    embs = []
    
    user_manager = UserManager(mongo_db=mongo_fa_db)
    
    fetched_user_ids = user_manager.get_users_id()
    
    for user_id in fetched_user_ids:
    
        user_face_embedings = []
        
        #load all images 
        image_filenames =[]
        dir_list = os.listdir(os.path.join(raw_folder_path, user_id))
            
        image_filenames = [f for f in dir_list if f.endswith(".jpeg") or f.endswith(".jpg")]
            
        for im_file in image_filenames:
                
            processor = ImageProcessor(src=os.path.join(raw_folder_path, user_id,im_file), output_folder=os.path.join(prepped_folder_path, user_id), keep_ratio=False)

            if processor.found_face_data is not None:
                
                face_embedding = facenet_model.embeddings(processor.get_reshaped_dims())
                
                user_face_embedings.append(face_embedding[0])
        
        labels.extend([user_id] * len(user_face_embedings))
        
        embs.append(user_face_embedings)
    
    embs = np.concatenate(embs)

    le = LabelEncoder().fit(labels)
    
    y = le.transform(labels)
    
    classifier = SVC(kernel='linear', probability=True).fit(embs, y)
    
    return le, classifier
        

# predicting
le, classifier = trainer()

test_emb = prepz_test_frame()



joblib.dump(le, f"{os.path.join(os.getcwd(), "svm_model")}\\le.gz")
joblib.dump(classifier, f"{os.path.join(os.getcwd(), "svm_model")}\\classifier.gz")

# le = joblib.load(f"{os.path.join(os.getcwd(), "svm_model")}\\le.gz")
# classifier = joblib.load(f"{os.path.join(os.getcwd(), "svm_model")}\\classifier.gz")

# prediction = le.inverse_transform(classifier.predict(test_emb))

# print(prediction)