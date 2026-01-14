
from pydantic import BaseModel
from typing import List, Optional

class TextCheckRequest(BaseModel):
    text: str

class MediaUrlRequest(BaseModel):
    url: str

class QuitPlanPredictRequest(BaseModel):
    features: list[float]

class TextToSpeechRequest(BaseModel):
    text: str


class DiaryLogRequest(BaseModel):
    id: Optional[int] = None
    date: str

    # Smoking status
    have_smoked: int
    cigarettes_smoked: int
    estimated_nicotine_intake: float

    # Mental & Feeling
    mood_level: int
    anxiety_level: int
    craving_level: int
    confidence_level: int

    # IoT & Health Data
    is_connect_iotdevice: int
    heart_rate: int
    spo2: int
    steps: int
    sleep_duration: float

    # NRT (Nicotine Replacement Therapy)
    is_use_nrt: int
    money_spent_on_nrt: float

    # Others
    reduction_percentage: float
    triggers: Optional[List[str]] = None
    note: Optional[str] = None
class SummaryRequest(BaseModel):
    member_name: str
    logs: List[DiaryLogRequest]