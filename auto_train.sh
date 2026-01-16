# Configuration
PROJECT_DIR="/home/luan-tran/projects/SmartQuitIoT/SmartQuitIoT-AI"
LOG_FILE="$PROJECT_DIR/auto_train.log"
API_URL="http://127.0.0.1:8000/train-models"
PYTHON_EXEC="$PROJECT_DIR/.venv/bin/python"
TRAINING_SCRIPT="app/services/ai_training_service.py"

echo "----------------------------------------------------------------" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Starting scheduled training..." >> "$LOG_FILE"

cd "$PROJECT_DIR" || {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] Failed to cd to $PROJECT_DIR" >> "$LOG_FILE"
    exit 1
}

# --- METHOD 1: Call API (Best for Hot Reload) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Triggering API training..." >> "$LOG_FILE"
response=$(curl -s -X POST "$API_URL")

if [[ $response == *"status"* ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] API Trigger Successful: $response" >> "$LOG_FILE"
else
    # --- METHOD 2: Fallback to running script manually if API is down ---
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] API unreachable. Running script manually..." >> "$LOG_FILE"
    $PYTHON_EXEC "$TRAINING_SCRIPT" >> "$LOG_FILE" 2>&1
fi

# Validation
MODEL_DIR="app/models"
SUCCESS_MODEL="$MODEL_DIR/smartquit_success_model.onnx"
CRAVING_MODEL="$MODEL_DIR/smartquit_craving_time_model.onnx"

if [ -f "$SUCCESS_MODEL" ] && [ -f "$CRAVING_MODEL" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] SUCCESS: Both models verified on disk." >> "$LOG_FILE"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] One or more models missing after training." >> "$LOG_FILE"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Task completed." >> "$LOG_FILE"
echo "----------------------------------------------------------------" >> "$LOG_FILE"