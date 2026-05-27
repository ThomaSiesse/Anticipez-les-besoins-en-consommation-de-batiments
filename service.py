import bentoml
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

# Charger le modèle
model_ref = bentoml.sklearn.get("seattle_ghg_model:latest")

# Schéma de validation Pydantic
class BuildingInput(BaseModel):
    surface_total_sqft: float = Field(..., gt=0, description="Total building surface in square feet")
    connected_to_gas: bool = Field(..., description="Is the building connected to natural gas?")
    primary_use_type: str = Field(..., description="Primary use type of the building")
    energy_star_score: float = Field(..., ge=1, le=100, description="ENERGY STAR score (1-100)")
    connected_to_steam: bool = Field(..., description="Is the building connected to Seattle's steam network?")
    number_of_floors: int = Field(..., gt=0, description="Number of floors")
    year_built: int = Field(..., ge=1900, le=2026, description="Year the building was constructed (1900-2026)")
    parking_surface_sqft: float = Field(..., ge=0, description="Parking surface in square feet")
    neighborhood: str = Field(..., description="Seattle neighborhood")

    @field_validator('surface_total_sqft')
    def surface_raisonnable(cls, v):
        if v > 10_000_000:
            raise ValueError("Surface too large — incoherent value")
        return v

    @field_validator('number_of_floors')
    def etages_raisonnables(cls, v):
        if v > 200:
            raise ValueError("Number of floors too high — incoherent value")
        return v

    @field_validator('primary_use_type')
    def type_usage_valide(cls, v):
        valid_types = [
            'Office', 'Warehouse', 'K-12 School', 'Large Office',
            'Mixed Use Property', 'Retail Store', 'Hotel', 'Worship Facility',
            'Distribution Center', 'Supermarket / Grocery Store', 'Medical Office',
            'Other'
        ]
        if v not in valid_types:
            raise ValueError(f"Invalid use type. Must be one of: {valid_types}")
        return v

    @field_validator('neighborhood')
    def quartier_valide(cls, v):
        valid_neighborhoods = [
            'DOWNTOWN', 'GREATER DUWAMISH', 'MAGNOLIA / QUEEN ANNE',
            'LAKE UNION', 'NORTHEAST', 'EAST', 'NORTHWEST', 'BALLARD',
            'NORTH', 'CENTRAL', 'SOUTHEAST', 'DELRIDGE', 'SOUTHWEST',
            'DELRIDGE NEIGHBORHOODS'
        ]
        v = v.upper().strip()
        if v not in valid_neighborhoods:
            raise ValueError(f"Invalid neighborhood. Must be one of: {valid_neighborhoods}")
        return v

@bentoml.service()
class SeattleGHGPredictor:
    def __init__(self):
        self.model = bentoml.sklearn.load_model("seattle_ghg_model:latest")
        self.enc = model_ref.custom_objects["encoder"]
        self.feature_names = model_ref.custom_objects["feature_names"]

    @bentoml.api()
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
            "emissions_co2_tonnes": round(float(prediction), 2),
            "message": f"Estimated CO2 emissions: {round(float(prediction), 2)} tonnes/year"
        }