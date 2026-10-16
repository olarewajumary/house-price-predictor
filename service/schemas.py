from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    overall_qual: int = Field(ge=1, le=10)
    gr_liv_area: int = Field(ge=300, le=6000)
    total_bsmt_sf: int = Field(ge=0, le=6000)
    garage_cars: int = Field(ge=0, le=4)
    year_built: int = Field(ge=1870, le=2026)
    full_bath: int = Field(ge=0, le=4)
    neighborhood: str
    exter_qual: str