PROJECT_DIR="/home/luan-tran/projects/SmartQuitIoT/SmartQuitIoT-AI"
LOG_FILE="$PROJECT_DIR/auto_train.log"
PYTHON_EXEC="$PROJECT_DIR/.venv/bin/python"
TRAINING_SCRIPT="app/services/ai_training.py"


echo "----------------------------------------------------------------" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Starting scheduled training task..." >> "$LOG_FILE"


cd "$PROJECT_DIR" || {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] Failed to change directory to $PROJECT_DIR" >> "$LOG_FILE"
    exit 1
}


echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Executing script: $TRAINING_SCRIPT" >> "$LOG_FILE"
$PYTHON_EXEC "$TRAINING_SCRIPT" >> "$LOG_FILE" 2>&1


MODEL_PATH="app/models/smartquit_model.onnx"

if [ -f "$MODEL_PATH" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] SUCCESS: Model validated at $MODEL_PATH" >> "$LOG_FILE"

else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] Model file was NOT found at $MODEL_PATH after training." >> "$LOG_FILE"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Task completed." >> "$LOG_FILE"
echo "----------------------------------------------------------------" >> "$LOG_FILE"