


from bson.objectid import ObjectId
from bson import json_util

import json

class UserManager(object):
    
    mongo_db = None
    
    users_collection = None
    
    def __init__(self, mongo_db):
        """
        :param mongo_db: selected database from mongodb client
        """
       
        self.mongo_db = mongo_db
        
        self.users_collection = mongo_db["users"]
        
        
    
    def create_user(self, user_data):
        """
        :param user_data: {_id: string, username: string}
        
        :return: user id
        """
        return self.users_collection.insert_one(user_data).inserted_id
    
    def get_user(self, user_id: str | None = None):
        """
        :return: {_id: string, username: string}
        """
        
        if user_id is None:
            return self.users_collection.find_one()["username"]
        else:
            return self.users_collection.find_one({"_id": ObjectId(user_id)})["username"]
    
    def get_users_id(self):
        """
        :return: [user id as strings]
        """
        
        output_list = []
        
        queried_users = self.users_collection.find()
        
        for user in queried_users:
            output_list.append(str(user["_id"]))
        
        return output_list
    
    def delete(self, user_id):
        
        pass




class EmbeddingManager(object):
    
    mongo_db = None
    mongo_emb_collection = None
    
    def __init__(self, mongo_db= None):
        
        self.mongo_db = mongo_db
        self.mongo_emb_collection = self.mongo_db["embeddings"]
        
        pass
    
    
    def create_embedding(self, ie: dict):
        """
        
        :param ie: {user_id: string, embeddings: [[]]}
        """
        print(ie)
        self.mongo_emb_collection.insert_one({"user_id": ie["user_id"], "embeddings": ie["embeddings"]})
    
    def read_embedding(self, user_id: str):
        """
        :param user_id: stringified version of the _id objectId from users collection
        :return : {user_id: str, embeddings: [[numbers]]}
        """
        
        fetched_emb = self.mongo_emb_collection.find({"user_id": user_id}, {"user_id": 1, "embeddings": 1 }) # as 
        
        output_emb = [emb["embeddings"] for emb in fetched_emb]
        
        
        return output_emb
    
    def read_all_embedding(self): # not proven indispensible yet
        pass
    
    def delete_embedding(self, user_id): #not proven absolutely neccesary yet
        pass