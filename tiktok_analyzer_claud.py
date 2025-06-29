import os
import base64
from moviepy.editor import VideoFileClip
import whisper
import cv2
import PIL.Image
from io import BytesIO
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()
os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY") 

client = Anthropic()


OUTPUT_DIR = "video_analysis_output"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
LLM_RESPONSES_DIR = os.path.join(OUTPUT_DIR, "llm_responses")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(LLM_RESPONSES_DIR, exist_ok=True)



def extract_audio(video_path):
    """Extracts audio from a video file using moviepy."""
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
    """Transcribes audio using the local OpenAI Whisper model."""
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

def extract_key_frames_for_claude(video_path, num_frames=5):
    """
    Extracts a few evenly spaced key frames from a video,
    converts them to JPEG format in memory, and returns them as PIL Images.
    Claude's API generally prefers common image formats like JPEG.
    """
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

def pil_image_to_base64(pil_image, image_format="jpeg"):
    """Converts a PIL Image object to a Base64 encoded string."""
    buffered = BytesIO()
    
    pil_image.save(buffered, format=image_format.upper())
    return base64.b64encode(buffered.getvalue()).decode("utf-8")



def analyze_video_with_claude(video_path):
    """Analyzes video frames and transcript using Claude 3 Sonnet."""
    print(f"\n--- Analyzing '{os.path.basename(video_path)}' with Claude ---")

    
    audio_path = extract_audio(video_path)
    transcript = transcribe_audio_whisper(audio_path)

   
    key_frames_pil = extract_key_frames_for_claude(video_path, num_frames=5)

    if not key_frames_pil:
        print("No frames extracted for Claude analysis.")
        return {
            "transcript": transcript,
            "description": "Could not analyze video due to frame extraction failure."
        }

    
    messages_content = []

   
    for frame_image_pil in key_frames_pil:
        base64_image = pil_image_to_base64(frame_image_pil, image_format="jpeg") 
        messages_content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg", #
                    "data": base64_image,
                },
            }
        )
    
    
    prompt_text = f"""
    Based on the provided images (key frames from a video) and the audio transcript, please describe the video's content.
    
    Overall Goal: Analyze this TikTok video about anti-vaccination for its multimodal features to understand its messaging, emotional appeal, and potential persuasive techniques.

    Instructions:
    Please analyze the provided video key frames and audio transcript. For each section below, extract specific details related to the video's content, focusing on elements that contribute to its anti-vaccination message. If a feature is not clearly present, state "Not explicitly clear" or "Absent."

    ### Video Analysis: [Video Filename]

    #### 1. Text-Based Features:
    * **Core Claims/Arguments:**
    * **Narrative Type:**
    * **Keywords and Phrases:**
    * **Source Credibility/References (explicit or implied):**
    * **Call to Action (CTA):**
    * **Emotional Language Used:**
    * **Misinformation/Disinformation Flags (if present):**

    #### 2. Image/Video-Based Features:
    * **Visual Setting/Environment:**
    * **Characters/Presenters:**
        * **Identity/Appearance:**
        * **Demographics (approximate):**
        * **Facial Expressions/Body Language:**
    * **Objects/Props Displayed:**
    * **On-Screen Text/Graphics:**
    * **Visual Tone/Aesthetics:**
    * **Audience Engagement Cues (visual):**

    #### 3. Audio-Based Features:
    * **Speaker Tone/Emotion:**
    * **Pacing and Emphasis:**
    * **Background Sounds/Music:**
    * **Vocal Characteristics:**
    * **Use of Pauses (strategic):**

    Here is the audio transcript:
    ---
    {transcript}
    ---
    """
    messages_content.append({"type": "text", "text": prompt_text})

    try:
        print("Sending frames and transcript to Claude API...")
        message = client.messages.create(
            model="claude-sonnet-4-20250514", 
            max_tokens=500, 
            messages=[
                {
                    "role": "user",
                    "content": messages_content,
                }
            ],
        )
        response_text = message.content[0].text 
        print("Claude analysis complete.")
        
        return {
            "transcript": transcript,
            "description": response_text
        }
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        if "AuthenticationError" in str(e) or "Invalid API Key" in str(e):
             print("Please ensure your ANTHROPIC_API_KEY environment variable is correctly set and is valid.")
             print("You might also need to have some credit on your Anthropic account for API usage.")
        return {
            "transcript": transcript,
            "description": f"Claude API analysis failed: {e}"
        }

# --- Main execution ---
if __name__ == "__main__":
    tiktok_videos = [
        "tiktok_video_3.mp4",
        "tiktok_video_4.mp4",
        "tiktok_video_5.mp4",
        "tiktok_video_6.mp4",
        
    ]

    
    claude_output_filename = os.path.join(LLM_RESPONSES_DIR, "claude_responses_medical_video.txt")

    
    with open(claude_output_filename, "w", encoding="utf-8") as f:
        f.write(f"--- Claude LLM Analysis Results ({os.path.basename(claude_output_filename)}) ---\n\n")

    for video_file in tiktok_videos:
        if not os.path.exists(video_file):
            print(f"Error: Video file '{video_file}' not found. Skipping.")
            continue

        claude_results = analyze_video_with_claude(video_file)

        
        print(f"\n{'='*50}\nRESULTS FOR: {os.path.basename(video_file)} (Claude)\n{'='*50}")
        print("\n--- Audio Transcript (Whisper) ---")
        print(claude_results["transcript"])
        print("\n--- Claude AI Description ---")
        print(claude_results["description"])
        print(f"\n{'='*50}\n")

        
        with open(claude_output_filename, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"VIDEO: {os.path.basename(video_file)}\n")
            f.write(f"{'='*60}\n\n")
            f.write("--- Audio Transcript (Whisper) ---\n")
            f.write(claude_results["transcript"] + "\n\n")
            f.write("--- Claude AI Description ---\n")
            f.write(claude_results["description"] + "\n")
            f.write(f"\n{'='*60}\n\n")

    print(f"\nAll Claude results saved to: {claude_output_filename}")