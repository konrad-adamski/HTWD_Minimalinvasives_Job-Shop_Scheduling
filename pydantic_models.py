from pydantic import BaseModel
from typing import Optional, List

class RoutingModel(BaseModel):
    id: int
    instance_id: int

    model_config = {
        "from_attributes": True
    }

class InstanceModel(BaseModel):
    id: int
    name: str
    routings: Optional[List[RoutingModel]] = None

    model_config = {
        "from_attributes": True
    }