from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta, timezone

load_dotenv()
c = MongoClient(os.getenv('MONGODB_URI'))
col = c['aqi_db']['features']

# Get records sorted by time
docs = list(col.find(
    {}, 
    {'timestamp':1, 'aqi':1, 'target_aqi_24h':1}
).sort('timestamp', -1).limit(100))

print("Timestamp            | Actual AQI | 24h Target stored")
print("-" * 60)
for doc in docs[:20]:
    ts = doc['timestamp'].strftime('%Y-%m-%d %H:%M')
    aqi = doc.get('aqi', '?')
    target = doc.get('target_aqi_24h', '?')
    print(f"{ts} | {aqi:<10} | {target}")

c.close()