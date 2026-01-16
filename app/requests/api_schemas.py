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


class DiaryAnalysisRequest(BaseModel):
    anxiety_level: Optional[int] = 0
    craving_level: Optional[int] = 0
    mood_level: Optional[int] = 5
    have_smoked: bool = False
    note: Optional[str] = ""
    triggers: Optional[List[str]] = []


class ReportChartRequest(BaseModel):
    member_name: str
    logs: List[dict]
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class CoachDataRequest(BaseModel):
    coach_name: str
    total_active_members: int
    appointments_completed: int
    member_highlights: List[str]
    top_triggers: List[str]


class PeakCravingRequest(BaseModel):
    age: int
    gender_code: int  # 1 for Male, 0 for Female
    ftnd_score: float
    smoke_avg_per_day: float
    mood_level: float
    anxiety_level: float

    # Predict for a specific day (0=Monday, 6=Sunday)
    # If not provided, backend uses today
    day_of_week: Optional[int] = None
