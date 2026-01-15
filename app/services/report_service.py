import matplotlib

matplotlib.use('Agg')  # Server mode
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64


class ReportService:
    def __init__(self):
        pass

    def generate_weekly_report_image(self, logs: list, member_name: str) -> str:
        if not logs:
            return None

        try:
            df = pd.DataFrame(logs)

            if 'date' not in df.columns: return None
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df['date_str'] = df['date'].dt.strftime('%d/%m')

            num_records = len(df)

            fig_width = max(10, num_records * 0.6)


            if fig_width > 25: fig_width = 25


            cols_to_ensure = [
                'cigarettes_smoked', 'craving_level', 'mood_level',
                'anxiety_level', 'confidence_level', 'heart_rate',
                'sleep_duration', 'steps'
            ]
            for col in cols_to_ensure:
                if col not in df.columns:
                    df[col] = np.nan
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            sns.set_theme(style="whitegrid")
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(fig_width, 18), sharex=True)

            sns.barplot(x='date_str', y='cigarettes_smoked', data=df, ax=ax1,
                        color='#e74c3c', alpha=0.7, label='Cigarettes')
            ax1.set_ylabel('Cigarettes', color='#c0392b', fontweight='bold')
            ax1.legend(loc='upper left')

            for i, val in enumerate(df['cigarettes_smoked']):
                if pd.isna(val):
                    ax1.text(i, 0, "N/A", ha='center', va='bottom', fontsize=10, color='gray', fontstyle='italic')

            ax1_twin = ax1.twinx()
            sns.lineplot(x='date_str', y='craving_level', data=df, ax=ax1_twin,
                         color='#2980b9', marker='o', linewidth=3, label='Craving (1-10)')
            ax1_twin.set_ylabel('Craving Level', color='#2980b9', fontweight='bold')
            ax1_twin.set_ylim(0, 11)
            ax1.set_title(f"Smoking Behavior & Cravings - {member_name}", fontsize=14, fontweight='bold')

            ax1.tick_params(axis='x', labelbottom=True, rotation=45)
            ax1.set_xlabel('')

            sns.lineplot(x='date_str', y='mood_level', data=df, ax=ax2,
                         color='#27ae60', marker='^', label='Mood')
            sns.lineplot(x='date_str', y='anxiety_level', data=df, ax=ax2,
                         color='#8e44ad', marker='X', linestyle='--', label='Anxiety')
            sns.lineplot(x='date_str', y='confidence_level', data=df, ax=ax2,
                         color='#f1c40f', marker='o', linestyle='-.', label='Confidence')

            ax2.set_ylabel('Score (1-10)', fontweight='bold')
            ax2.set_ylim(0, 11)
            ax2.legend(loc='upper right')
            ax2.set_title("Psychological Trends", fontsize=14, fontweight='bold')

            ax2.tick_params(axis='x', labelbottom=True, rotation=45)
            ax2.set_xlabel('')

            sns.barplot(x='date_str', y='sleep_duration', data=df, ax=ax3,
                        color='#34495e', alpha=0.5, label='Sleep (hours)')
            ax3.set_ylabel('Sleep (Hours)', color='#34495e', fontweight='bold')
            ax3.legend(loc='upper left')

            for i, val in enumerate(df['sleep_duration']):
                if pd.isna(val):
                    ax3.text(i, 0, "N/A", ha='center', va='bottom', fontsize=10, color='gray', fontstyle='italic')

            ax3_twin = ax3.twinx()
            sns.lineplot(x='date_str', y='heart_rate', data=df, ax=ax3_twin,
                         color='#e67e22', marker='d', linewidth=2, label='Avg Heart Rate')
            ax3_twin.set_ylabel('BPM', color='#d35400', fontweight='bold')

            valid_hr = df['heart_rate'].dropna()
            if not valid_hr.empty and valid_hr.max() > 0:
                ax3_twin.set_ylim(valid_hr.min() * 0.9, valid_hr.max() * 1.1)
            else:
                ax3_twin.set_ylim(0, 100)

            ax3.set_title("Health & IoT Metrics", fontsize=14, fontweight='bold')
            ax3.set_xlabel('Date', fontweight='bold')

            ax3.tick_params(axis='x', rotation=45)

            plt.subplots_adjust(hspace=0.6, bottom=0.1)
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return image_base64

        except Exception as e:
            print(f"Error generating advanced chart: {e}")
            return None


report_service = ReportService()