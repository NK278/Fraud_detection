import os
import sys

import certifi
import pymongo

from src.constant import *
from src.exception import CustomException

ca = certifi.where()


class MongoDBClient:
    client = None

    def __init__(self, database_name='Fraud_Detection') -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv("mongodb+srv://nishchalgaur2003:zatfem-wymxyX-kyvqy3@cluster0.kvvgt6j.mongodb.net/?retryWrites=true&w=majority")
                if mongo_db_url is None:
                    raise Exception("Environment key: MONGO_DB_URL is not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as e:
            raise CustomException(e, sys)
