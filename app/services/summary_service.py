import re
from app.models import summary_model, summary_tokenizer


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


def generate_coach_summary(member_name: str, logs: list) -> str:

    if not logs:
        return "No data available."

    if summary_model is None or summary_tokenizer is None:
        return "AI Model is not loaded on server."

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
        "You are an expert smoking cessation coach assistant. "
        "Analyze the weekly logs provided below. "
        "Your Task: Write a professional summary for the human Coach.\n"
        "Focus on these key aspects:\n"
        "1. Smoking Status & Progress (Success/Relapse).\n"
        "2. Physical Health (Heart Rate, SpO2, Sleep, Steps impact).\n"
        "3. Psychological State (Relationship between Mood/Anxiety/Craving and Triggers).\n"
        "4. NRT Usage & Financial efficiency (CRITICAL: If logs say 'DID NOT USE NRT', explicitly state that NO NRT was used. Do NOT hallucinate that they used it).\n"
        "5. Actionable Advice (MUST INCLUDE: Specific recommendation based on logs)\n"
        "IMPORTANT: Use Markdown formatting with bold headers for readability. Keep it concise."
    )

    user_message = f"Member Name: {member_name}\n\nWEEKLY LOGS:\n{log_text}\n\nPlease generate the summary report."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    try:
        text = summary_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = summary_tokenizer([text], return_tensors="pt").to(summary_model.device)

        generated_ids = summary_model.generate(
            **model_inputs,
            max_new_tokens=600,
            temperature=0.6,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = summary_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    except Exception as e:
        print(f"Summary Generation Error: {str(e)}")
        return f"AI Processing Error: {str(e)}"


class SummaryService:
    def __init__(self):
        pass


summary_service = SummaryService()