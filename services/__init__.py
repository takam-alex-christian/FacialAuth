
from flask_pymongo import PyMongo

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
            return json_util.dumps(self.users_collection.find_one())
        else:
            return json_util.dumps(self.users_collection.find_one({"_id": ObjectId(user_id)}))
    
    def delete(self, user_id):
        
        pass
    