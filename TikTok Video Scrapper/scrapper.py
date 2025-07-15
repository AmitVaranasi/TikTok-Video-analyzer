import yt_dlp
import json
import os
import time
import random
import pandas as pd # Import the pandas library

# --- Configuration ---
excel_file_path = "URLs.xlsx"  # <<< IMPORTANT: Change this to your Excel file name
url_column_name = "URLs"           # <<< IMPORTANT: Change this to the exact name of your column containing the URLs

output_folder = "tiktok_downloads"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")

ydl_opts = {
    'format': 'best',
    'outtmpl': os.path.join(output_folder, '%(id)s.%(ext)s'),
    'writedescription': True,
    'writeinfojson': True,
    'noplaylist': True, # Ensure only single videos are downloaded, not playlists if a URL happens to be a playlist
    'ignoreerrors': True, # Continue downloading even if one video fails
}

# --- Read URLs from Excel ---
try:
    df = pd.read_excel(excel_file_path)
    if url_column_name not in df.columns:
        raise ValueError(f"Column '{url_column_name}' not found in the Excel file.")
    
    # Extract URLs and filter out any empty or non-string values
    tiktok_urls = df[url_column_name].dropna().astype(str).tolist()
    print(f"Found {len(tiktok_urls)} URLs in '{excel_file_path}' from column '{url_column_name}'.")

except FileNotFoundError:
    print(f"Error: Excel file not found at '{excel_file_path}'")
    exit()
except ValueError as ve:
    print(f"Error reading Excel file: {ve}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while reading the Excel file: {e}")
    exit()

# --- Download Videos ---
for i, url in enumerate(tiktok_urls):
    print(f"\n--- Processing URL {i+1}/{len(tiktok_urls)}: {url} ---")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            video_id = info.get('id', 'N/A')
            title = info.get('title', 'N/A')
            author = info.get('uploader', 'N/A')
            print(f"Successfully downloaded '{title}' by {author} (ID: {video_id})")

    except Exception as e:
        print(f"Failed to download {url}. Error: {e}")
    
    # Add a delay for ethical/technical reasons
    if i < len(tiktok_urls) - 1: # Don't sleep after the last download
        sleep_time = random.uniform(2, 6) # Random delay between 2 and 6 seconds
        print(f"Waiting for {sleep_time:.2f} seconds before next download...")
        time.sleep(sleep_time)

print("\nAll downloads complete.")
print(f"Videos and metadata saved in: {os.path.abspath(output_folder)}")