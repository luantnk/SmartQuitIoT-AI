import re
import json
from app.models import hf_client

HF_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def clean_java_triggers(raw_trigger):
    if not raw_trigger:
        return "None"

    if isinstance(raw_trigger, list):
        return ", ".join(raw_trigger)

    if isinstance(raw_trigger, str):
        if len(raw_trigger) < 50 and "java" not in raw_trigger:
            return raw_trigger

        matches = re.findall(r'\b[A-Z][a-zA-Z]+\b', raw_trigger)
        cleaned = [w for w in matches if w not in ["ArrayList", "java", "util"]]

        if cleaned:
            return ", ".join(cleaned)

    return "None"


def generate_coach_summary(member_name: str, logs: list) -> dict:
    if not logs:

        return {
            "summary": "No data available.",
            "risk_level": "UNKNOWN",
            "alerts": [],
            "status_color": "gray"
        }

    if hf_client is None:
        return {
            "summary": "AI Model (Hugging Face Client) is not loaded on server.",
            "risk_level": "ERROR",
            "alerts": ["System Error"],
            "status_color": "red"
        }

    log_text = ""
    for log in logs:
        triggers_raw = log.get("triggers")
        triggers = clean_java_triggers(triggers_raw)

        nrt_status = "Yes" if log.get('is_use_nrt') == 1 else "No"
        money_spent = log.get('money_spent_on_nrt', 0)

        hr = log.get('heart_rate', 0)
        spo2 = log.get('spo2', 0)
        steps = log.get('steps', 0)
        sleep = log.get('sleep_duration', 0)

        health_info = ""
        if log.get('is_connect_iotdevice') == 1:
            health_info = (
                f" | [Health Data]: HR {hr} bpm, "
                f"SpO2 {spo2}%, "
                f"Sleep {sleep}h, "
                f"Steps {steps}"
            )

        confidence = log.get('confidence_level', 0)
        craving = log.get('craving_level', 0)
        mood = log.get('mood_level', 0)
        anxiety = log.get('anxiety_level', 0)

        log_text += (
            f"- Date {log['date']}: "
            f"Smoked {log['cigarettes_smoked']} cigs. "
            f"Craving {craving}/10, Confidence {confidence}/10. "
            f"Mood {mood}/10, Anxiety {anxiety}/10. "
            f"NRT: {nrt_status} (${money_spent}). "
            f"Triggers: {triggers}. "
            f"{health_info}. "
            f"Note: {log.get('note', '').strip()}\n"
        )



    system_prompt = (
        "You are an expert smoking cessation clinical coach (SmartQuitIoT). "
        "Your goal is to analyze the logs and return a valid **JSON Object**.\n\n"

        "**STRICT JSON OUTPUT STRUCTURE:**\n"
        "{\n"
        "  \"summary_data\": \"(String) The Clinical Insight Report in Markdown...\",\n"
        "  \"risk_level\": \"(String) CRITICAL | HIGH | MEDIUM | LOW\",\n"
        "  \"alerts\": [\"(Array of strings) Short tags e.g. 'No NRT', 'HR Spike'\"],\n"
        "  \"status_color\": \"(String) red | yellow | green\"\n"
        "}\n\n"

        "**CONTENT RULES FOR THE 'summary' FIELD:**\n"
        "1. **Analysis Over Listing**: DO NOT simply list dates or raw numbers. Instead, ANALYZE the relationship (e.g., 'Smoking volume spiked by 60% following severe insomnia').\n"
        "2. **Thematic Structure**: Organize the report into three clear narrative sections:\n"
        "   - **Section 1: Behavioral Dynamics**: Focus on progress, relapse patterns, and the 'Why' behind the smoking.\n"
        "   - **Section 2: Biopsychosocial Correlation**: Connect IoT data (Sleep, HR) with psychological states (Anxiety, Mood).\n"
        "   - **Section 3: Intervention Strategy**: Provide 2-3 professional, direct clinical commands.\n"
        "3. **NRT Integrity**: (CRITICAL) If NRT was not used, highlight this as a failure. Do not hallucinate usage.\n"
        "4. **Tone**: Professional, analytical, and urgent where necessary.\n"
        "5. **Formatting**: Use Markdown with `###` for headers. Use bold text for key insights. Keep concise."
    )

    user_message = f"Member Name: {member_name}\n\nWEEKLY LOGS:\n{log_text}\n\nPlease generate the JSON response."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    try:
        response = hf_client.chat.completions.create(
            model=HF_MODEL_ID,
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        return json.loads(content)

    except json.JSONDecodeError:
        return {
            "summary": response.choices[0].message.content,
            "risk_level": "UNKNOWN",
            "alerts": ["JSON Parse Error"],
            "status_color": "gray"
        }
    except Exception as e:
        print(f"HF API Error: {str(e)}")
        return {
            "summary": f"Error generating report: {str(e)}",
            "risk_level": "ERROR",
            "alerts": ["System Error"],
            "status_color": "red"
        }


class SummaryService:
    def __init__(self):
        pass


summary_service = SummaryService()