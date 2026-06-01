import requests

API_KEY = "670c5f5780e3f5da454f3ffa1ce100e8435c992f3a9b1d10a879da793d6ef881"

# Check sensors for the most promising locations
location_ids = [8156, 1894633, 4515644, 4791924]

for loc_id in location_ids:
    r = requests.get(
        f"https://api.openaq.org/v3/locations/{loc_id}/sensors",
        headers={"X-API-Key": API_KEY}
    )
    data = r.json()
    print(f"\nLocation {loc_id}:")
    for sensor in data.get("results", []):
        param = sensor.get("parameter", {})
        coverage = sensor.get("coverage", {})
        print(f"  Sensor {sensor['id']} | {param.get('name')} ({param.get('units')}) | "
              f"from {coverage.get('datetimeFrom', {}).get('local', 'N/A')[:10]} "
              f"to {coverage.get('datetimeTo', {}).get('local', 'N/A')[:10]} | "
              f"count: {coverage.get('observedCount', 0)}")