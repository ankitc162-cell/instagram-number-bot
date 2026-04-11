"""
Daily Instagram Reel Generator & Poster
- Fetches follower count from Instagram Graph API
- Generates number facts using Google Gemini (free)
- Creates voiceover using Microsoft Edge TTS (free, no API key)
- Fetches relevant video clips from Pexels (free API)
- Assembles engaging reel: video clips + big number overlay + voiceover
- Posts reel to Instagram
"""

import os
import time
import asyncio
import textwrap
import requests
import tempfile
import random
import json
import edge_tts
from google import genai
from pathlib import Path
from datetime import datetime

from moviepy.editor import (
    VideoFileClip, ColorClip, TextClip, CompositeVideoClip,
    AudioFileClip, concatenate_videoclips, ImageClip
)
from moviepy.video.fx.all import crop, resize

# ── Config from environment ───────────────────────────────────────────────────
INSTAGRAM_ACCESS_TOKEN = os.environ["INSTAGRAM_ACCESS_TOKEN"]
INSTAGRAM_USER_ID      = os.environ["INSTAGRAM_USER_ID"]
GEMINI_API_KEY         = os.environ["GEMINI_API_KEY"]
PEXELS_API_KEY         = os.environ["PEXELS_API_KEY"]

EDGE_TTS_VOICE = "en-US-GuyNeural"   # energetic male voice
DAY_NUMBER     = int(os.environ.get("DAY_NUMBER", 1))

REEL_W, REEL_H = 1080, 1920


# ─────────────────────────────────────────────────────────────────────────────
# 1. Get follower count
# ─────────────────────────────────────────────────────────────────────────────
def get_follower_count() -> int:
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
# 2. Generate script + search keywords with Gemini
# ─────────────────────────────────────────────────────────────────────────────
def generate_script_and_keywords(day: int, followers: int) -> dict:
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""You are a scriptwriter for a viral educational Instagram Reels page.

Today is Day {day}. The page has {followers} followers.

Write a voiceover script about the number {followers} AND provide Pexels video search keywords for each fact.

Return ONLY valid JSON in this exact format:
{{
  "script": "Full voiceover script here...",
  "segments": [
    {{"fact": "One sentence summary of fact 1", "keywords": "pexels search term for fact 1"}},
    {{"fact": "One sentence summary of fact 2", "keywords": "pexels search term for fact 2"}},
    {{"fact": "One sentence summary of fact 3", "keywords": "pexels search term for fact 3"}}
  ]
}}

Script rules:
- Start with: "Day {day} of posting facts about the number {followers}. Today we have {followers} followers."
- Include 3 fascinating facts about the number {followers}
- End with: "Follow us and be part of this journey — all the way to 1 million!"
- 150-200 words total, punchy and enthusiastic

Keywords rules:
- Each keyword should be a simple 1-3 word Pexels search term related to the fact topic
- Examples: "space galaxy", "ancient egypt", "mathematics", "ocean waves", "city lights"
- Make them visually interesting and likely to return good footage"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    data = json.loads(text)
    print(f"[INFO] Script generated ({len(data['script'].split())} words)")
    print(f"[INFO] Segments: {[s['keywords'] for s in data['segments']]}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fetch video clips from Pexels
# ─────────────────────────────────────────────────────────────────────────────
def fetch_pexels_video(query: str, output_path: str) -> str:
    headers = {"Authorization": PEXELS_API_KEY}
    params  = {"query": query, "per_page": 10, "orientation": "portrait", "size": "medium"}

    resp = requests.get("https://api.pexels.com/videos/search",
                        headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    videos = resp.json().get("videos", [])

    if not videos:
        # fallback to landscape if no portrait
        params["orientation"] = "landscape"
        resp = requests.get("https://api.pexels.com/videos/search",
                            headers=headers, params=params, timeout=15)
        videos = resp.json().get("videos", [])

    if not videos:
        print(f"[WARN] No videos found for '{query}', using color fallback")
        return None

    # Pick a random video from top results for variety
    video = random.choice(videos[:5])

    # Get the best quality video file (HD preferred)
    video_files = sorted(video["video_files"],
                         key=lambda x: x.get("width", 0), reverse=True)
    # Pick highest res that's not 4K (too large)
    chosen = next((v for v in video_files if v.get("width", 0) <= 1920), video_files[0])

    video_url = chosen["link"]
    print(f"[INFO] Downloading Pexels clip: {query} → {video_url[:60]}...")

    video_resp = requests.get(video_url, timeout=60, stream=True)
    video_resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in video_resp.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"[INFO] Clip saved → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 4. Generate voiceover with Edge TTS
# ─────────────────────────────────────────────────────────────────────────────
async def _edge_tts_generate(script: str, output_path: str):
    communicate = edge_tts.Communicate(script, EDGE_TTS_VOICE)
    await communicate.save(output_path)

def generate_voiceover(script: str, output_path: str = "voiceover.mp3") -> str:
    asyncio.run(_edge_tts_generate(script, output_path))
    print(f"[INFO] Voiceover saved → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 5. Build the reel video
# ─────────────────────────────────────────────────────────────────────────────
def make_number_overlay(number: int, day: int, duration: float):
    """Big number + day badge that sits on top of all clips."""
    number_clip = (
        TextClip(str(number), fontsize=200, color="white",
                 font="DejaVu-Sans-Bold", stroke_color="black", stroke_width=4)
        .set_duration(duration)
        .set_position(("center", 0.35), relative=True)
    )
    day_clip = (
        TextClip(f"DAY {day}", fontsize=60, color="#FFC800",
                 font="DejaVu-Sans-Bold", stroke_color="black", stroke_width=2)
        .set_duration(duration)
        .set_position(("center", 0.22), relative=True)
    )
    return [day_clip, number_clip]


def crop_to_portrait(clip):
    """Crop any clip to 9:16 portrait (1080x1920)."""
    target_ratio = REEL_W / REEL_H
    clip_ratio   = clip.w / clip.h

    if clip_ratio > target_ratio:
        # wider than needed — crop sides
        new_w = int(clip.h * target_ratio)
        clip  = crop(clip, width=new_w, x_center=clip.w / 2)
    else:
        # taller than needed — crop top/bottom
        new_h = int(clip.w / target_ratio)
        clip  = crop(clip, height=new_h, y_center=clip.h / 2)

    return resize(clip, (REEL_W, REEL_H))


def build_video(data: dict, audio_path: str,
                day: int, followers: int,
                output_path: str = "reel.mp4") -> str:

    audio    = AudioFileClip(audio_path)
    duration = audio.duration
    segments = data["segments"]
    n        = len(segments)

    # Divide audio duration equally among segments
    seg_duration = duration / n
    print(f"[INFO] Total duration: {duration:.1f}s, {n} segments × {seg_duration:.1f}s each")

    clips = []
    for i, seg in enumerate(segments):
        clip_path = f"clip_{i}.mp4"
        pexels_path = fetch_pexels_video(seg["keywords"], clip_path)

        if pexels_path:
            try:
                raw = VideoFileClip(pexels_path)
                # Loop if clip is shorter than segment duration
                if raw.duration < seg_duration:
                    loops = int(seg_duration / raw.duration) + 1
                    from moviepy.editor import concatenate_videoclips as cv
                    raw = cv([raw] * loops)
                raw = raw.subclip(0, seg_duration)
                raw = crop_to_portrait(raw).without_audio()
            except Exception as e:
                print(f"[WARN] Failed to process clip {i}: {e}, using color fallback")
                raw = ColorClip(size=(REEL_W, REEL_H),
                                color=(15, 15, 30)).set_duration(seg_duration)
        else:
            raw = ColorClip(size=(REEL_W, REEL_H),
                            color=(15, 15, 30)).set_duration(seg_duration)

        # Dark overlay for readability
        overlay = ColorClip(size=(REEL_W, REEL_H),
                            color=(0, 0, 0)).set_opacity(0.45).set_duration(seg_duration)

        clips.append(CompositeVideoClip([raw, overlay], size=(REEL_W, REEL_H)))

    # Concatenate all segments
    base_video = concatenate_videoclips(clips, method="compose")

    # Number + day overlay on top of everything
    overlays = make_number_overlay(followers, day, duration)

    final = CompositeVideoClip(
        [base_video] + overlays,
        size=(REEL_W, REEL_H)
    ).set_audio(audio).set_duration(duration)

    final.write_videofile(
        output_path, fps=30, codec="libx264", audio_codec="aac",
        temp_audiofile="temp_audio.m4a", remove_temp=True, logger=None
    )
    print(f"[INFO] Final video built → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 6. Upload & post to Instagram
# ─────────────────────────────────────────────────────────────────────────────
def upload_video_to_hosting(video_path: str) -> str:
    with open(video_path, "rb") as f:
        resp = requests.post(
            f"https://transfer.sh/{Path(video_path).name}",
            data=f,
            headers={"Max-Downloads": "5", "Max-Days": "1"},
            timeout=180
        )
    resp.raise_for_status()
    url = resp.text.strip()
    print(f"[INFO] Video uploaded → {url}")
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
            raise RuntimeError(f"Instagram processing failed: {status_resp.json()}")
    else:
        raise TimeoutError("Instagram timed out after 5 minutes.")

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
# 7. Orchestrate
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"  Daily Reel Generator -- {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}\n")

    followers  = get_follower_count()
    data       = generate_script_and_keywords(DAY_NUMBER, followers)
    audio_path = generate_voiceover(data["script"])
    video_path = build_video(data, audio_path, DAY_NUMBER, followers)
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
