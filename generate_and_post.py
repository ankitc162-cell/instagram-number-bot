"""
Daily Instagram Reel Generator & Poster
- Fetches follower count from Instagram Graph API
- Generates number facts using Google Gemini (free)
- Creates voiceover using ElevenLabs (free tier)
- Assembles video using MoviePy
- Posts reel to Instagram
"""

import os
import time
import textwrap
import requests
from google import genai
from pathlib import Path
from datetime import datetime

# Optional heavy deps
try:
    from moviepy.editor import (
        ColorClip, TextClip, CompositeVideoClip, AudioFileClip
    )
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

# ── Config from environment ───────────────────────────────────────────────────
INSTAGRAM_ACCESS_TOKEN = os.environ["INSTAGRAM_ACCESS_TOKEN"]
INSTAGRAM_USER_ID      = os.environ["INSTAGRAM_USER_ID"]
ELEVENLABS_API_KEY     = os.environ["ELEVENLABS_API_KEY"]
GEMINI_API_KEY         = os.environ["GEMINI_API_KEY"]

ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
DAY_NUMBER = int(os.environ.get("DAY_NUMBER", 1))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Get follower count (via Facebook Graph API)
# ─────────────────────────────────────────────────────────────────────────────
def get_follower_count() -> int:
    # Use Facebook Graph API (not graph.instagram.com)
    url = (
        f"https://graph.facebook.com/v19.0/{INSTAGRAM_USER_ID}"
        f"?fields=followers_count&access_token={INSTAGRAM_ACCESS_TOKEN}"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    count = resp.json().get("followers_count", 0)
    print(f"[INFO] Follower count: {count}")
    return count


# ─────────────────────────────────────────────────────────────────────────────
# 2. Generate script with Google Gemini (FREE)
# ─────────────────────────────────────────────────────────────────────────────
def generate_script(day: int, followers: int) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""You are a scriptwriter for a viral educational Instagram Reels page.

Today is Day {day}. The page currently has {followers} followers.

Write a short, engaging voiceover script for a reel about the number {followers}.

Rules:
- Start with exactly: "Day {day} of posting facts about the number {followers}. Today we have {followers} followers."
- Include 3 to 4 fascinating, surprising facts about the number {followers}. Choose the most compelling count — pick facts that will genuinely wow a general audience.
- Facts can be mathematical, historical, cultural, scientific, or from pop culture — whatever is most mind-blowing for THIS specific number.
- Keep each fact to 1-2 short punchy sentences. No filler words.
- End with exactly: "Follow us and be part of this journey — all the way to 1 million!"
- Total script length: 60-90 seconds when spoken aloud (~150-220 words).
- Tone: enthusiastic, clear, like a knowledgeable friend — NOT a stiff documentary.
- Output ONLY the voiceover script. No stage directions, no titles, no extra commentary."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    script = response.text.strip()
    print(f"[INFO] Script generated ({len(script.split())} words)")
    return script


# ─────────────────────────────────────────────────────────────────────────────
# 3. Generate voiceover with ElevenLabs
# ─────────────────────────────────────────────────────────────────────────────
def generate_voiceover(script: str, output_path: str = "voiceover.mp3") -> str:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": script,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.4,
            "use_speaker_boost": True
        }
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)
    print(f"[INFO] Voiceover saved -> {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build the video (vertical 9:16 reel)
# ─────────────────────────────────────────────────────────────────────────────
REEL_W, REEL_H = 1080, 1920
BG_COLOR   = (15, 15, 30)
ACCENT     = (255, 200, 0)
TEXT_WHITE = (255, 255, 255)

def _text_clip(text, fontsize, color, duration, pos, max_width=900):
    wrapped = "\n".join(textwrap.wrap(text, width=max(10, max_width // (fontsize // 2))))
    return (
        TextClip(wrapped, fontsize=fontsize, color=color,
                 font="DejaVu-Sans-Bold", method="caption", size=(max_width, None))
        .set_duration(duration)
        .set_position(pos)
    )

def build_video(script: str, audio_path: str,
                day: int, followers: int,
                output_path: str = "reel.mp4") -> str:
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError("moviepy not installed.")

    audio    = AudioFileClip(audio_path)
    duration = audio.duration

    bg          = ColorClip(size=(REEL_W, REEL_H), color=BG_COLOR).set_duration(duration)
    number_clip = _text_clip(str(followers), 220, list(ACCENT),  duration, ("center", 480))
    day_clip    = _text_clip(f"DAY {day}",   72,  list(ACCENT),  duration, ("center", 280))
    sub_clip    = _text_clip("Facts about this number", 52, list(TEXT_WHITE), duration, ("center", 740))
    cta_clip    = _text_clip("Follow -> 1,000,000", 58, list(ACCENT), duration, ("center", REEL_H - 180))

    video = CompositeVideoClip(
        [bg, number_clip, day_clip, sub_clip, cta_clip],
        size=(REEL_W, REEL_H)
    ).set_audio(audio)

    video.write_videofile(
        output_path, fps=30, codec="libx264", audio_codec="aac",
        temp_audiofile="temp_audio.m4a", remove_temp=True, logger=None
    )
    print(f"[INFO] Video built -> {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 5. Upload & publish reel to Instagram
# ─────────────────────────────────────────────────────────────────────────────
def upload_video_to_hosting(video_path: str) -> str:
    with open(video_path, "rb") as f:
        resp = requests.post(
            f"https://transfer.sh/{Path(video_path).name}",
            data=f,
            headers={"Max-Downloads": "5", "Max-Days": "1"},
            timeout=120
        )
    resp.raise_for_status()
    url = resp.text.strip()
    print(f"[INFO] Video uploaded -> {url}")
    return url


def post_reel(video_url: str, caption: str) -> str:
    base = "https://graph.facebook.com/v19.0"

    container_resp = requests.post(
        f"{base}/{INSTAGRAM_USER_ID}/media",
        data={
            "media_type":   "REELS",
            "video_url":    video_url,
            "caption":      caption,
            "access_token": INSTAGRAM_ACCESS_TOKEN,
        },
        timeout=30,
    )
    container_resp.raise_for_status()
    container_id = container_resp.json()["id"]
    print(f"[INFO] Media container created: {container_id}")

    for attempt in range(30):
        time.sleep(10)
        status_resp = requests.get(
            f"{base}/{container_id}",
            params={"fields": "status_code,status", "access_token": INSTAGRAM_ACCESS_TOKEN},
            timeout=15,
        )
        status_resp.raise_for_status()
        status = status_resp.json().get("status_code", "")
        print(f"[INFO] Container status ({attempt+1}/30): {status}")
        if status == "FINISHED":
            break
        if status == "ERROR":
            raise RuntimeError(f"Instagram video processing failed: {status_resp.json()}")
    else:
        raise TimeoutError("Instagram video processing timed out after 5 minutes.")

    publish_resp = requests.post(
        f"{base}/{INSTAGRAM_USER_ID}/media_publish",
        data={"creation_id": container_id, "access_token": INSTAGRAM_ACCESS_TOKEN},
        timeout=30,
    )
    publish_resp.raise_for_status()
    media_id = publish_resp.json()["id"]
    print(f"[INFO] Reel published! Media ID: {media_id}")
    return media_id


# ─────────────────────────────────────────────────────────────────────────────
# 6. Orchestrate
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"  Daily Reel Generator -- {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}\n")

    followers = get_follower_count()
    script    = generate_script(DAY_NUMBER, followers)

    print("\n--- SCRIPT PREVIEW ---")
    print(script)
    print("----------------------\n")

    audio_path = generate_voiceover(script)
    video_path = build_video(script, audio_path, DAY_NUMBER, followers)
    video_url  = upload_video_to_hosting(video_path)

    caption = (
        f"Day {DAY_NUMBER} of posting facts about the number {followers}\n\n"
        f"Today we have {followers} followers and THIS number is fascinating!\n\n"
        f"Follow us and be part of this journey to 1,000,000\n\n"
        f"#numberfacts #didyouknow #educational #facts #math "
        f"#instagramreels #viral #mindblown #science"
    )

    post_reel(video_url, caption)
    print("\nDone! Reel posted successfully.")


if __name__ == "__main__":
    main()
