# TikTok Video Multimodal Analysis

This project demonstrates the capability of advanced Large Language Models (LLMs) to analyze TikTok videos by combining visual (keyframes) and auditory (transcript) information. It extracts audio, transcribes it, takes key visual frames, and then sends this multimodal data to various LLM APIs (Google Gemini, Anthropic Claude, and OpenAI ChatGPT/GPT-4o) to generate detailed descriptions of the video content.

## Features

* **Audio Extraction:** Extracts audio tracks from MP4 video files.
* **Audio Transcription:** Transcribes the extracted audio into text using OpenAI's Whisper model (local execution).
* **Key Frame Extraction:** Selects a configurable number of evenly spaced key frames from the video.
* **Multimodal LLM Analysis:**
    * **Google Gemini:** Utilizes `gemini-1.5-flash` (or `gemini-2.0-flash` if available and preferred) for video description.
    * **Anthropic Claude:** Uses `claude-3-5-sonnet-20240620` (Claude 3.5 Sonnet) for detailed analysis.
    * **OpenAI ChatGPT:** Integrates with `gpt-4o` (GPT-4o) for comprehensive multimodal understanding.
* **Structured Output:** Saves the audio transcript and LLM-generated descriptions into separate, organized text files for each LLM, allowing for easy comparison.

## Prerequisites

Before running the project, ensure you have the following installed:

* **Python:** Version 3.9 or higher.
* **FFmpeg:** A powerful multimedia framework required by `moviepy` for video and audio processing.
    * **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your system's PATH.
    * **macOS:** Install via Homebrew: `brew install ffmpeg`
    * **Linux:** Install via your package manager: `sudo apt-get update && sudo apt-get install ffmpeg` (Debian/Ubuntu) or `sudo dnf install ffmpeg` (Fedora).
* **API Keys:**
    * Google Gemini API Key
    * Anthropic Claude API Key
    * OpenAI API Key

## Setup and Installation

1.  **Clone the Repository (or create project directory):**
    If this is a new directory, create it and save the Python scripts (e.g., `tiktok_analyzer_gemini.py`, `tiktok_analyzer_claude.py`, `tiktok_analyzer_openai.py` and a combined main script if you have one) into it.

2.  **Create a Virtual Environment (Recommended):**
    It's good practice to use a virtual environment to manage project dependencies.

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    .\venv\Scripts\activate   # On Windows
    ```

3.  **Install Required Python Packages:**

    ```bash
    pip install moviepy opencv-python "openai-whisper[torch]" google-generativeai anthropic openai
    ```
    * `moviepy`: For video and audio processing.
    * `opencv-python`: For image processing (frame extraction).
    * `openai-whisper[torch]`: For local audio transcription. `[torch]` ensures PyTorch is installed for Whisper's models.
    * `google-generativeai`: Google Gemini API client.
    * `anthropic`: Anthropic Claude API client.
    * `openai`: OpenAI API client (for GPT-4o).

## Configuration (API Keys)

For security, it is **highly recommended** to set your API keys as environment variables.

* **Google Gemini:** `GOOGLE_API_KEY`
* **Anthropic Claude:** `ANTHROPIC_API_KEY`
* **OpenAI:** `OPENAI_API_KEY`

**How to set Environment Variables:**
## Configuration (API Keys via .env file)

To securely manage your API keys, create a file named `.env` in the root directory of your project (the same directory where your Python scripts are located).

**`.env` file example:**

GOOGLE_API_KEY='YOUR_GEMINI_API_KEY'
ANTHROPIC_API_KEY='YOUR_CLAUDE_API_KEY'
OPENAI_API_KEY='YOUR_OPENAI_API_KEY'


**Important Security Note:**
**NEVER** commit your `.env` file to version control (e.g., Git). Add `.env` to your `.gitignore` file to prevent accidentally pushing your sensitive keys to public repositories.

**Integrating `.env` into your Python scripts:**

At the very top of each Python script (`tiktok_analyzer_gemini.py`, `tiktok_analyzer_claude.py`, `tiktok_analyzer_openai.py`), add the following two lines of code:

```python
from dotenv import load_dotenv
load_dotenv()
This will automatically load the variables from your .env file into your script's environment. You can then access them using os.getenv('YOUR_VARIABLE_NAME'). The API clients (e.g., google.generativeai, anthropic, openai) are designed to automatically pick up these environment variables if they are set correctly.

## How to Run

1.  **Place your TikTok videos** (e.g., `tiktok_video_1.mp4`, `tiktok_video_2.mp4`) in the same directory as your Python script(s), or update the `tiktok_videos` list in the script to point to their correct paths.

2.  **Execute the main analysis script(s).** If you have separate scripts for each LLM, run them individually:

    ```bash
    python3 tiktok_analyzer.py
    python3 tiktok_analyzer_claud.py
    python3 tiktok_analyzer_openai.py
    ```
    *(If you have a single orchestrating script, adjust the command accordingly.)*

The script will:
* Create an `video_analysis_output` directory if it doesn't exist.
* Extract audio to `video_analysis_output/audio/`.
* Extract frames to `video_analysis_output/frames/`.
* Save LLM responses to `video_analysis_output/llm_responses/`.

## Output Files

The results of the analysis will be saved in the `video_analysis_output/llm_responses/` directory:

* `gemini_responses.txt`: Contains audio transcripts and descriptions generated by Google Gemini.
* `claude_responses.txt`: Contains audio transcripts and descriptions generated by Anthropic Claude.
* `openai_responses.txt`: Contains audio transcripts and descriptions generated by OpenAI's GPT-4o.

Each file will append results for every video processed, clearly separated by video filename.

## Project Structure

.
├── tiktok_video_1.mp4
├── tiktok_video_2.mp4
├── tiktok_analyzer_gemini.py
├── tiktok_analyzer_claude.py
├── tiktok_analyzer_openai.py
├── venv/                       # Python Virtual Environment (created by python3 -m venv venv)
└── video_analysis_output/
├── audio/                  # Extracted audio files (.mp3)
├── frames/                 # Extracted keyframes (if saved, though not strictly needed for LLM input)
└── llm_responses/
├── gemini_responses.txt
├── claude_responses.txt
└── openai_responses.txt
