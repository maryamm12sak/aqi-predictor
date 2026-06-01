import requests

TOKEN = "b8c601f40e46dd93e2e2ddf7bb98d55b8e72f12f"

# Search for stations near Karachi using lat/lng bounds
r = requests.get(
    f"https://api.waqi.info/map/bounds/?latlng=24.6,66.7,25.1,67.5&token={TOKEN}",
    timeout=10
).json()

print(f"Found {len(r.get('data', []))} stations near Karachi:\n")
for s in r.get("data", []):
    uid = s.get("uid")
    aqi = s.get("aqi")
    name = s.get("station", {}).get("name")
    time = s.get("station", {}).get("time")
    print(f"UID: @{uid} | {name} | AQI: {aqi} | Time: {time}")