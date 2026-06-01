import requests

API_KEY = "670c5f5780e3f5da454f3ffa1ce100e8435c992f3a9b1d10a879da793d6ef881"

# Find Karachi stations
r = requests.get(
    "https://api.openaq.org/v3/locations",
    params={"coordinates": "24.8607,67.0011", "radius": 25000, "limit": 10},
    headers={"X-API-Key": API_KEY}
)
print("Status:", r.status_code)
data = r.json()
print("Found", len(data.get("results", [])), "locations")
for loc in data.get("results", []):
    print(f"ID: {loc['id']} | Name: {loc['name']} | Sensors: {len(loc.get('sensors', []))}")