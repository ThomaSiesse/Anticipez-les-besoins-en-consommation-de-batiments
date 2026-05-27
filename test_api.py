import requests

data = {
    "input_data": {
        "surface_total_sqft": 50000,
        "connected_to_gas": True,
        "primary_use_type": "Hotel",
        "energy_star_score": 75,
        "connected_to_steam": False,
        "number_of_floors": 10,
        "year_built": 1995,
        "parking_surface_sqft": 5000,
        "neighborhood": "DOWNTOWN"
    }
}

response = requests.post("http://localhost:3000/predict", json=data)
print(response.json())