import os
import base64
from moviepy.editor import VideoFileClip
import whisper
import cv2
import PIL.Image
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") 
client = OpenAI()


OUTPUT_DIR = "video_analysis_output"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")

LLM_RESPONSES_DIR = os.path.join(OUTPUT_DIR, "llm_responses")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(LLM_RESPONSES_DIR, exist_ok=True) 


def extract_audio(video_path):
    audio_path = os.path.join(AUDIO_DIR, os.path.basename(video_path).replace(".mp4", ".mp3"))
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path)
        audio_clip.close()
        video_clip.close()
        print(f"Audio extracted to: {audio_path}")
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def transcribe_audio_whisper(audio_path):
    if not audio_path:
        return "No audio available for transcription."
    print(f"Transcribing audio with Whisper: {audio_path}...")
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        transcript = result["text"]
        print("Audio transcription complete.")
        return transcript
    except Exception as e:
        print(f"Error transcribing audio with Whisper: {e}")
        return "Audio transcription failed."

def extract_key_frames_for_openai(video_path, num_frames=5):
    frames_pil = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        print(f"Video {video_path} has no frames.")
        return []
    step = max(1, frame_count // num_frames)
    print(f"Extracting {num_frames} key frames from: {video_path}...")
    for i in range(num_frames):
        frame_idx = min(i * step, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = PIL.Image.fromarray(rgb_image)
            frames_pil.append(pil_image)
        else:
            print(f"Warning: Could not read frame {frame_idx}.")
            break
    cap.release()
    print(f"Extracted {len(frames_pil)} frames.")
    return frames_pil

def pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def analyze_video_with_openai(video_path):
    print(f"\n--- Analyzing '{os.path.basename(video_path)}' with OpenAI ---")
    audio_path = extract_audio(video_path)
    transcript = transcribe_audio_whisper(audio_path)
    key_frames_pil = extract_key_frames_for_openai(video_path, num_frames=5)

    if not key_frames_pil:
        print("No frames extracted for OpenAI analysis.")
        return {
            "transcript": transcript,
            "description": "Could not analyze video due to frame extraction failure."
        }

    messages_content = []
    for frame_image_pil in key_frames_pil:
        base64_image = pil_image_to_base64(frame_image_pil)
        messages_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                },
            }
        )
    
    prompt_text = """

    **Role:** You are an expert social media content analyst specializing in health communication and misinformation detection. Your task is to extract specific, quantifiable features from TikTok videos based on visual (keyframes) and auditory (transcript) information.

**Instruction:**
Analyze the provided video content (keyframes and audio transcript) thoroughly. Your goal is to identify and categorize the salient features listed below.
**You MUST output your analysis as a single JSON object.**
Adhere strictly to the provided JSON structure, including all keys and their specified data types (boolean, string, array of strings).
For categorical features, choose *exactly one* from the provided options. If a feature is not applicable or clearly discernible, use `null` for strings/categories or `false` for boolean flags.

**Input Data:**
* **Video Key Frames:** [Will be provided as image inputs to the LLM]
* **Audio Transcript:**
    ```
    {transcript}
    ```

**Output JSON Schema / Structure to Adhere To:**

```json
{
  "video_filename": "string",
  "analysis_timestamp": "string (YYYY-MM-DDTHH:MM:SSZ)",
  "features": {
    "text_based": {
      "narrative_type": "string (Personal Anecdote | Scientific 'Debunking' (Pseudoscientific) | Conspiracy Theory | Call to Action | News/Reportage | Product Promotion | Other | Unclear)",
      "core_claims_keywords": ["array of strings (e.g., 'vaccine harm', 'government conspiracy', 'natural immunity superiority', 'vaccine ineffective', 'ingredient scare')"],
      "source_cited_type": "string (Personal Experience | Pseudoscientific Expert | Unnamed 'Studies' | Mainstream Media (critically) | Anti-Vax Org | Absent | Unclear)",
      "cta_type": ["array of strings (e.g., 'share_video', 'do_own_research', 'join_group', 'refuse_vaccine', 'attend_event', 'other_cta', 'no_cta')"],
      "emotional_language_present": {
        "fear": "boolean",
        "anger": "boolean",
        "treachery_betrayal": "boolean",
        "urgency": "boolean",
        "hope_empowerment": "boolean",
        "sadness_distress": "boolean",
        "humor_sarcasm": "boolean"
      },
      "misinformation_flags": ["array of strings (e.g., 'unsubstantiated causal link', 'distrust in medical professionals', 'fear-mongering re: ingredients', 'hidden truth narrative', 'cherry-picked data', 'ad_hominem', 'straw_man', 'false_dilemma', 'slippery_slope', 'appeal_to_emotion', 'absent')"]
    },
    "image_video_based": {
      "setting_type": "string (Indoor_Home | Outdoor_Public_Space | Official_Medical_Setting | Studio_Professional | Abstract_Graphics_Only | Unclear)",
      "presenter_cues": {
        "appears_medical_professional": "boolean",
        "appears_official_setting": "boolean",
        "presents_documentation": "boolean"
      },
      "presenter_facial_expression": "string (Neutral_Calm | Distressed_Sad | Angry_Frustrated | Enthusiastic_Confident | Other_Mixed | Unclear)",
      "presenter_body_language_intensity": "string (Calm_Subtle | Moderate_Gestures | Energetic_Agitated_Gestures | Unclear)",
      "prop_presence": {
        "medical_items_shown": "boolean",
        "personal_items_shown": "boolean",
        "protest_signs_shown": "boolean"
      },
      "on_screen_text_type": "string (Call_to_Action | Warning_Alarm | Conspiracy_Keyword | Personal_Message | Data_Statistics | Other | Absent)",
      "visual_tone_aesthetics": {
        "visual_quality": "string (High_Production | Medium_Production | Low_Amateur | Grainy_Distorted | Graphics_Only)",
        "dominant_color_palette": "string (Warm | Cool | Neutral | High_Contrast | Dark_Somber | Mixed | Unclear)",
        "tiktok_specific_filters_effects": "boolean"
      },
      "audience_engagement_visual_cues": ["array of strings (e.g., 'direct_camera_address', 'on_screen_text_emphasis', 'visual_call_to_action', 'emojis_overlays', 'duet_stitch_potential', 'absent')"]
    },
    "audio_based": {
      "background_music_presence": "boolean",
      "background_music_mood": "string (Ominous_Suspenseful | Sad_Melancholic | Uplifting_Hopeful | Neutral_Ambient | Energetic_Aggressive | Absent | Unclear)",
      "sound_effects_presence": "boolean",
      "sound_effects_type": "string (Alarms_Sirens | Dramatic_Thuds_Swooshes | Cheering_Applause | Other | Absent | Unclear)",
      "speaker_tone_emotion": "string (Neutral | Calm | Urgent | Angry | Distressed | Conspiratorial | Sarcastic | Sympathetic | Other_Mixed | Unclear)",
      "speaker_volume_variation": "string (Consistent | Moderate_Variation | High_Variation_for_Emphasis | Unclear)",
      "strategic_pauses_used": "boolean"
    }
  }
}

    """
    messages_content.append({"type": "text", "text": prompt_text})

    try:
        print("Sending frames and transcript to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": messages_content,
                }
            ],
            max_tokens=500,
        )
        response_text = response.choices[0].message.content
        print("OpenAI analysis complete.")
        
        return {
            "transcript": transcript,
            "description": response_text
        }
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        if "AuthenticationError" in str(e) or "Invalid API key" in str(e):
             print("Please ensure your OPENAI_API_KEY environment variable is correctly set and is valid.")
             print("You might also need to have some credit on your OpenAI account for API usage.")
        return {
            "transcript": transcript,
            "description": f"OpenAI API analysis failed: {e}"
        }

# --- Main execution ---
if __name__ == "__main__":
    tiktok_videos = [
        "./TikTok Videos/tiktok_video_3.mp4",
        "./TikTok Videos/tiktok_video_4.mp4",
        "./TikTok Videos/tiktok_video_5.mp4",
        "./TikTok Videos/tiktok_video_6.mp4",
        
    ]

   
    openai_output_filename = os.path.join(LLM_RESPONSES_DIR, "openai_responses_medical_video.txt")

    
    with open(openai_output_filename, "w", encoding="utf-8") as f:
        f.write(f"--- OpenAI LLM Analysis Results ({os.path.basename(openai_output_filename)}) ---\n\n")

    for video_file in tiktok_videos:
        if not os.path.exists(video_file):
            print(f"Error: Video file '{video_file}' not found. Skipping.")
            continue

        openai_results = analyze_video_with_openai(video_file)

      
        print(f"\n{'='*50}\nRESULTS FOR: {os.path.basename(video_file)} (OpenAI)\n{'='*50}")
        print("\n--- Audio Transcript (Whisper) ---")
        print(openai_results["transcript"])
        print("\n--- OpenAI AI Description ---")
        print(openai_results["description"])
        print(f"\n{'='*50}\n")

        
        with open(openai_output_filename, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"VIDEO: {os.path.basename(video_file)}\n")
            f.write(f"{'='*60}\n\n")
            f.write("--- Audio Transcript (Whisper) ---\n")
            f.write(openai_results["transcript"] + "\n\n")
            f.write("--- OpenAI AI Description ---\n")
            f.write(openai_results["description"] + "\n")
            f.write(f"\n{'='*60}\n\n")

    print(f"\nAll OpenAI results saved to: {openai_output_filename}")