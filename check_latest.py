from pymongo import MongoClient
from dotenv import load_dotenv
import os
load_dotenv()
c = MongoClient(os.getenv('MONGODB_URI'))
doc = c['aqi_db']['features'].find_one(sort=[('timestamp',-1)])
print('Latest record:')
for k,v in doc.items():
    if k != '_id':
        print(f'  {k}: {v}')
c.close()