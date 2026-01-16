import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from datetime import datetime
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from dotenv import load_dotenv

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE_DIR))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

MODEL_DIR = os.path.join(PROJECT_ROOT, "app", "models")

MODEL_PATH_SUCCESS = os.path.join(MODEL_DIR, "smartquit_success_model.onnx")
MODEL_PATH_CRAVING = os.path.join(MODEL_DIR, "smartquit_craving_time_model.onnx")

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
    print(f"[ERROR] Missing env vars in .env")
    sys.exit(1)

db_connection_str = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
try:
    db_connection = create_engine(db_connection_str)
except Exception as e:
    print(f"[ERROR] Connection Error: {e}")
    sys.exit(1)


def load_full_rich_data():

    print("Loading FULL rich data from MariaDB...")

    sql_query = """
                SELECT
                    -- Diary Record
                    dr.anxiety_level,
                    dr.craving_level,
                    dr.mood_level,
                    dr.have_smoked,
                    dr.created_at as record_time,
                    dr.heart_rate,
                    dr.sleep_duration,

                    -- Member
                    m.gender,
                    m.dob,
                    m.morning_reminder_time,
                    m.time_zone,

                    -- Quit Plan & Phase
                    qp.ftnd_score,
                    qp.is_active,
                    p.progress,
                    p.status      as phase_status,

                    -- Form Metric
                    fm.smoke_avg_per_day,
                    fm.minutes_after_waking_to_smoke,
                    fm.number_of_years_of_smoking,
                    fm.cigarettes_per_package,
                    fm.money_per_package

                FROM diary_record dr
                         LEFT JOIN member m ON dr.member_id = m.id
                         LEFT JOIN quit_plan qp ON m.id = qp.member_id
                         LEFT JOIN form_metric fm ON fm.id = qp.form_metric_id
                         LEFT JOIN phase p ON p.quit_plan_id = qp.id
                WHERE dr.created_at IS NOT NULL \
                """

    try:
        df = pd.read_sql(sql_query, db_connection)
        if df.empty:
            print("[WARN] Database is empty.")
            return None
        print(f"Data loaded successfully. Rows: {len(df)}")
        return df
    except Exception as e:
        print(f"[ERROR] DB Query Error: {e}")
        return None


def preprocess_common_features(df):

    print("Preprocessing Data...")

    # Handle Binary/Bytes columns (MariaDB BIT fields often come as bytes)
    def parse_binary_col(val):
        if isinstance(val, bytes):
            return int.from_bytes(val, "big")
        return val

    if "have_smoked" in df.columns:
        df["have_smoked"] = df["have_smoked"].apply(parse_binary_col).fillna(0).astype(int)

    # Calculate Age
    current_year = datetime.now().year
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["age"] = current_year - df["dob"].dt.year
    df["age"] = df["age"].fillna(30)

    # Gender Encoding
    df["gender_code"] = df["gender"].apply(lambda x: 1 if str(x).upper() == "MALE" else 0)

    # Handle Nulls in Metrics
    cols_to_zero = ["anxiety_level", "craving_level", "heart_rate", "sleep_duration", "progress", "ftnd_score"]
    for col in cols_to_zero:
        df[col] = df[col].fillna(0).astype(float)

    df["mood_level"] = df["mood_level"].fillna(5).astype(float)
    df["smoke_avg_per_day"] = df["smoke_avg_per_day"].fillna(10).astype(float)

    return df


def train_peak_craving_time_model(df):
    print("\n--- Training Peak Craving Time Model ---")

    df["record_time"] = pd.to_datetime(df["record_time"])
    df["hour_of_day"] = df["record_time"].dt.hour
    df["day_of_week"] = df["record_time"].dt.dayofweek


    feature_cols = [
        "hour_of_day",
        "day_of_week",
        "ftnd_score",
        "smoke_avg_per_day",
        "age",
        "gender_code",
        "mood_level",
        "anxiety_level"
    ]

    training_data = df.dropna(subset=feature_cols + ['craving_level'])

    X = training_data[feature_cols]
    y = training_data['craving_level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        objective='reg:squarederror'
    )

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Craving Time Model R2 Score: {score:.4f}")

    initial_type = [("float_input", FloatTensorType([None, len(feature_cols)]))]
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    onnxmltools.utils.save_model(onnx_model, MODEL_PATH_CRAVING)
    print(f"SUCCESS: Craving Time Model saved at {MODEL_PATH_CRAVING}")
    print(f"Features used: {feature_cols}")


def train_success_model(df):

    print("\n--- Training Success/Failure Model ---")

    def define_success(row):
        if row["have_smoked"] > 0: return 0  # Failed
        if str(row["phase_status"]) == "COMPLETED": return 1  # Success
        return 1

    df["target_label"] = df.apply(define_success, axis=1)

    feature_cols = [
        "ftnd_score",
        "smoke_avg_per_day",
        "minutes_after_waking_to_smoke",
        "age",
        "gender_code",
        "anxiety_level",
        "craving_level",
        "mood_level",
        "heart_rate",
        "progress"
    ]

    X = df[feature_cols]
    y = df["target_label"]

    if len(np.unique(y)) < 2:
        print("[SKIP] Not enough variety in target labels to train classifier.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", use_label_encoder=False
    )

    model.fit(X_train, y_train)
    print(f"Success Model Accuracy: {model.score(X_test, y_test):.4f}")

    initial_type = [("float_input", FloatTensorType([None, len(feature_cols)]))]
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
    onnxmltools.utils.save_model(onnx_model, MODEL_PATH_SUCCESS)
    print(f"SUCCESS: Success Model saved at {MODEL_PATH_SUCCESS}")


if __name__ == "__main__":

    raw_df = load_full_rich_data()
    if raw_df is not None and not raw_df.empty:
        processed_df = preprocess_common_features(raw_df)

        train_success_model(processed_df)
        train_peak_craving_time_model(processed_df)