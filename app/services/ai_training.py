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


load_dotenv()


DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
    sys.exit(1)

db_connection_str = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

try:
    db_connection = create_engine(db_connection_str)
except Exception as e:

    sys.exit(1)


def load_rich_data():
    print("🔄 Loading rich data (Joining Member + Metric + Plan)...")

    sql_query = """
                SELECT m.id         as member_id,
                       m.dob,
                       m.gender,
                       p.start_date as plan_start_date,
                       p.cigarettes_per_package,
                       met.avg_anxiety,
                       met.avg_craving_level,
                       met.avg_mood,
                       met.current_craving_level,
                       met.streaks,
                       met.money_saved,
                       met.heart_rate,
                       met.smoke_free_day_percentage,
                       met.relapse_count_in_phase
                FROM member m
                         JOIN metric met ON m.id = met.member_id
                         LEFT JOIN plan p ON m.id = p.member_id
                WHERE p.status = 'IN_PROGRESS'
                   OR p.status = 'CANCELED' \
                """

    try:
        df = pd.read_sql(sql_query, db_connection)
        print(f"✅ Data loaded. Rows: {len(df)}")
        return df
    except Exception as e:
        print(f"❌ DB Error during query: {e}")
        return None


def preprocess_features(df):
    print("🛠 Feature Engineering...")

    current_year = datetime.now().year
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = current_year - df['dob'].dt.year

    df['plan_start_date'] = pd.to_datetime(df['plan_start_date'])
    df['days_on_plan'] = (datetime.now() - df['plan_start_date']).dt.days

    df['gender_code'] = df['gender'].apply(lambda x: 1 if str(x).upper() == 'MALE' else 0)

    feature_columns = [
        'age',
        'gender_code',
        'days_on_plan',
        'cigarettes_per_package',
        'avg_anxiety',
        'avg_craving_level',
        'current_craving_level',
        'streaks',
        'heart_rate',
        'smoke_free_day_percentage'
    ]


    df[feature_columns] = df[feature_columns].fillna(0)


    df['is_high_risk'] = df.apply(
        lambda row: 1 if (row['relapse_count_in_phase'] > 0 or row['current_craving_level'] > 7) else 0,
        axis=1
    )

    return df, feature_columns


def train_and_export_onnx(df, features):
    print("🚀 Training Model...")

    X = df[features]
    y = df['is_high_risk']

    if len(y.unique()) < 2:
        print("⚠️ CẢNH BÁO: Dữ liệu chỉ có 1 nhãn (toàn bộ High Risk hoặc toàn bộ Low Risk).")
        print("   Mô hình có thể không học được gì hữu ích.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test) if len(X_test) > 0 else 0
    print(f"✅ Training Done. Accuracy: {accuracy:.2f}")
    print("💾 Converting to ONNX format...")
    initial_type = [('float_input', FloatTensorType([None, len(features)]))]
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
    onnx_filename = "smartquit_risk_model.onnx"
    onnxmltools.utils.save_model(onnx_model, onnx_filename)
    print(f"🎉 Model saved as '{onnx_filename}'")
    print(f"📝 Feature Order for Java: {features}")
    return onnx_filename


if __name__ == "__main__":
    raw_df = load_rich_data()
    if raw_df is not None and not raw_df.empty:
        processed_df, feats = preprocess_features(raw_df)

        if len(processed_df) > 5:
            train_and_export_onnx(processed_df, feats)
        else:
            print("⚠️ Not enough data to train (Need > 5 rows).")
    else:
        print("⚠️ No data found in database.")