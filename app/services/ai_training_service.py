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
MODEL_PATH = os.path.join(MODEL_DIR, "smartquit_model.onnx")

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
    print(f"[ERROR] Missing env vars. Checked .env at: {PROJECT_ROOT}")
    sys.exit(1)

db_connection_str = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
try:
    db_connection = create_engine(db_connection_str)
except Exception as e:
    print(f"[ERROR] Connection Error: {e}")
    sys.exit(1)


def load_rich_data():
    print("Loading FULL rich data...")
    sql_query = """
                SELECT qp.ftnd_score, \
                       fm.smoke_avg_per_day, \
                       fm.minutes_after_waking_to_smoke, \
                       fm.number_of_years_of_smoking, \
                       m.gender, \
                       m.dob, \
                       dr.anxiety_level, \
                       dr.craving_level, \
                       dr.mood_level, \
                       dr.heart_rate, \
                       dr.sleep_duration, \
                       dr.have_smoked, \
                       p.progress, \
                       p.status as phase_status
                FROM quit_plan qp
                         JOIN member m ON qp.member_id = m.id
                         JOIN form_metric fm ON qp.form_metric_id = fm.id
                         LEFT JOIN phase p ON qp.id = p.quit_plan_id
                         LEFT JOIN diary_record dr ON m.id = dr.member_id
                WHERE qp.is_active = 1 \
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


def preprocess_features(df):
    print("Feature Engineering...")

    def parse_binary_col(val):
        if isinstance(val, bytes):
            return int.from_bytes(val, "big")
        return val

    if "have_smoked" in df.columns:
        df["have_smoked"] = (
            df["have_smoked"].apply(parse_binary_col).fillna(0).astype(int)
        )

    current_year = datetime.now().year
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["age"] = current_year - df["dob"].dt.year
    df["age"] = df["age"].fillna(25)

    df["gender_code"] = df["gender"].apply(
        lambda x: 1 if str(x).upper() == "MALE" else 0
    )

    # Fill NULL
    df["anxiety_level"] = df["anxiety_level"].fillna(0)
    df["craving_level"] = df["craving_level"].fillna(0)
    df["mood_level"] = df["mood_level"].fillna(5)
    df["heart_rate"] = df["heart_rate"].fillna(0)
    df["sleep_duration"] = df["sleep_duration"].fillna(0)
    df["progress"] = df["progress"].fillna(0)

    def define_success(row):
        if row["have_smoked"] > 0:
            return 0
        if str(row["phase_status"]) == "COMPLETED":
            return 1
        return 1

    df["target_label"] = df.apply(define_success, axis=1)

    feature_columns = [
        "ftnd_score",
        "smoke_avg_per_day",
        "minutes_after_waking_to_smoke",
        "age",
        "gender_code",
        "anxiety_level",
        "craving_level",
        "mood_level",
        "heart_rate",
        "sleep_duration",
        "progress",
    ]

    df[feature_columns] = df[feature_columns].astype(float)
    return df, feature_columns


def train_and_export_onnx(df, features):
    print("Training SmartQuit Predictive Model...")
    X = df[features]
    y = df["target_label"]

    print(f"Phân phối nhãn dữ liệu: {np.unique(y, return_counts=True)}")
    if len(np.unique(y)) < 2:
        print("[ERROR] CRITICAL: Dữ liệu chỉ có 1 loại nhãn.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    try:
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test) if not X_test.empty else 0
        print(f"Training Done. Accuracy: {accuracy:.4f}")

        print("Converting to ONNX...")
        initial_type = [("float_input", FloatTensorType([None, len(features)]))]
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            print(f"Created directory: {MODEL_DIR}")

        onnxmltools.utils.save_model(onnx_model, MODEL_PATH)
        print(f"SUCCESS: Model saved at {MODEL_PATH}")

    except Exception as e:
        print(f"[ERROR] Training Failed: {e}")


if __name__ == "__main__":
    raw_df = load_rich_data()
    if raw_df is not None and not raw_df.empty:
        processed_df, feats = preprocess_features(raw_df)
        if processed_df is not None:
            train_and_export_onnx(processed_df, feats)
