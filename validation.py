from pydantic import BaseModel, Field, field_validator

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
