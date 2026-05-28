import bentoml
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

# Charger le modèle
model_ref = bentoml.sklearn.get("seattle_ghg_model:latest")

from validation import BuildingInput

@bentoml.service()
class SeattleGHGPredictor:
    def __init__(self):
        self.model = bentoml.sklearn.load_model("seattle_ghg_model:latest")
        self.enc = model_ref.custom_objects["encoder"]
        self.feature_names = model_ref.custom_objects["feature_names"]

    @bentoml.api(route="/predict")
    def predict(self, input_data: BuildingInput) -> dict:
        surface_log = np.log1p(input_data.surface_total_sqft)
        cat_data = pd.DataFrame({
            'PrimaryPropertyType': [input_data.primary_use_type],
            'Neighborhood': [input_data.neighborhood.upper().strip()]
        })
        encoded = self.enc.transform(cat_data)
        encoded_cols = self.enc.get_feature_names_out(['PrimaryPropertyType', 'Neighborhood'])
        
        features = {
        'PropertyGFATotal_log': surface_log,
        'HasGas': int(input_data.connected_to_gas),
        'ENERGYSTARScore': input_data.energy_star_score,
        'HasSteam': int(input_data.connected_to_steam),
        'NumberofFloors': input_data.number_of_floors,
        'YearBuilt': input_data.year_built,
        'PropertyGFAParking': input_data.parking_surface_sqft,
        'HasParking': 1 if input_data.parking_surface_sqft > 0 else 0,
        'GFAperFloor': np.expm1(surface_log) / input_data.number_of_floors,
        }
        X = pd.DataFrame([features])
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols)
        X = pd.concat([X, encoded_df], axis=1)
        X = X.reindex(columns=self.feature_names, fill_value=0)
        prediction_log = self.model.predict(X.values)
        prediction = np.expm1(prediction_log[0])
        return {
            "message": f"Estimated CO2 emissions: {round(float(prediction), 2)} tonnes/year"
        }