from datetime import datetime
from pydantic import BaseModel


class CandleSchema(BaseModel):
    time: datetime
    low: float
    high: float
    open: float
    close: float
    volume: float

