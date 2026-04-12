"""
Daily Instagram Reel Generator & Poster
- Fetches follower count using instagrapi
- Generates number facts using Google Gemini (free)
- Creates voiceover using Microsoft Edge TTS (free)
- Fetches video clips from Pexels (free API)
- Assembles engaging reel: video clips + number overlay + voiceover
- Posts reel to Instagram using instagrapi
"""

import os
import time
import asyncio
import textwrap
import requests
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

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
PEXELS_API_KEY = os.environ["PEXELS_API_KEY"]
EDGE_TTS_VOICE = "hi-IN-MadhurNeural"
DAY_NUMBER     = int(os.environ.get("DAY_NUMBER", 1))
REEL_W, REEL_H = 1080, 1920


# ─────────────────────────────────────────────────────────────────────────────
# 1. Get follower count via instagrapi
# ─────────────────────────────────────────────────────────────────────────────
def get_ig_client():
    from instagrapi import Client
    ig_username  = os.environ["INSTAGRAM_USERNAME"]
    ig_password  = os.environ["INSTAGRAM_PASSWORD"]
    session_file = "ig_session.json"
    cl = Client()
    cl.delay_range = [1, 3]
    if os.path.exists(session_file):
        try:
            cl.load_settings(session_file)
            cl.login(ig_username, ig_password)
            cl.get_timeline_feed()
            print("[INFO] Logged in using saved session")
            return cl
        except Exception:
            print("[INFO] Session expired, logging in fresh...")
    cl = Client()
    cl.delay_range = [1, 3]
    cl.login(ig_username, ig_password)
    cl.dump_settings(session_file)
    print("[INFO] Logged in fresh")
    return cl


def get_follower_count() -> int:
    cl    = get_ig_client()
    user  = cl.user_info_by_username(os.environ["INSTAGRAM_USERNAME"])
    count = user.follower_count
    print(f"[INFO] Follower count: {count}")
    return count


# ─────────────────────────────────────────────────────────────────────────────
# 2. Generate script + keywords with Gemini
# ─────────────────────────────────────────────────────────────────────────────
def generate_script_and_keywords(day: int, followers: int) -> dict:
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        f"Aap ek viral educational Instagram Reels page ke liye scriptwriter hain.\n"
        f"Aaj Day {day} hai. Page par {followers} followers hain.\n\n"
        f"Number {followers} ke baare mein ek Hindi voiceover script likho AUR har fact ke liye Pexels video search keywords do.\n\n"
        f"Sirf valid JSON return karo is exact format mein:\n"
        '{{"script": "Poora voiceover script yahan...", "segments": [{{"fact": "Fact 1 ka ek sentence summary", "keywords": "pexels search term in English"}}, {{"fact": "Fact 2 ka ek sentence summary", "keywords": "pexels search term in English"}}, {{"fact": "Fact 3 ka ek sentence summary", "keywords": "pexels search term in English"}}]}}\n\n'
        f"Script rules:\n"
        f"- BILKUL EXACTLY is tarah shuru karo: Day {day} of posting facts about the number of my followers. Aaj mere {followers} followers hain.\n"
        f"- Phir excitement ke saath kaho jaise abhi kuch amazing discover kiya ho — is number ke baare mein yeh jaankar aap hairan ho jaoge!\n"
        f"- EXACTLY 3 fascinating facts likho number {followers} ke baare mein. Na zyada, na kam.\n"
        f"- Har fact 1-2 punchy conversational sentences mein ho. Seedha viewer se baat karo. Phrases use karo jaise: yeh sun ke aap hairan ho jaoge, yeh baat mera dimaag ghuma gayi, kya aap jaante hain, sochiye zaraa.\n"
        f"- Facts ke beech genuine wonder react karo — viewer ko feel ho ki aap dono saath mein discover kar rahe ho.\n"
        f"- Jahan relevant ho, India, cricket, Bollywood ya rozaana ki zindagi ka reference do.\n"
        f"- BILKUL EXACTLY is tarah khatam karo: Aapka follow kal ka number badal deta hai. Abhi follow karo aur chalo saath mein 1 million tak pahunche.\n"
        f"- 150-180 words total. Conversational, warm, genuine Hindi — textbook jaisi nahi.\n\n"
        f"Keywords rules:\n"
        f"- EXACTLY 3 segments do, ek har fact ke liye\n"
        f"- Keywords ENGLISH mein ho — simple 1-3 word Pexels search terms jo us fact se visually related ho"
    )
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            break
        except Exception as e:
            if attempt == 2:
                raise
            print(f"[WARN] Gemini attempt {attempt+1} failed: {e}, retrying in 30s...")
            time.sleep(30)
    text = response.text.strip()
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
def fetch_pexels_video(query: str, output_path: str):
    headers = {"Authorization": PEXELS_API_KEY}
    for orientation in ["portrait", "landscape"]:
        params = {"query": query, "per_page": 10, "orientation": orientation, "size": "medium"}
        resp   = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        videos = resp.json().get("videos", [])
        if videos:
            break
    if not videos:
        print(f"[WARN] No videos found for '{query}'")
        return None
    video      = random.choice(videos[:5])
    video_files = sorted(video["video_files"], key=lambda x: x.get("width", 0), reverse=True)
    chosen     = next((v for v in video_files if v.get("width", 0) <= 1920), video_files[0])
    print(f"[INFO] Downloading Pexels clip: {query} -> {chosen['link'][:60]}...")
    r = requests.get(chosen["link"], timeout=60, stream=True)
    r.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[INFO] Clip saved -> {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 4. Generate voiceover with Edge TTS
# ─────────────────────────────────────────────────────────────────────────────
async def _edge_tts_generate(script: str, output_path: str):
    communicate = edge_tts.Communicate(script, EDGE_TTS_VOICE)
    await communicate.save(output_path)

def generate_voiceover(script: str, output_path: str = "voiceover.mp3") -> str:
    asyncio.run(_edge_tts_generate(script, output_path))
    print(f"[INFO] Voiceover saved -> {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 5. Build the reel video
# ─────────────────────────────────────────────────────────────────────────────
def make_text_image(text, fontsize, color, stroke_color="black", stroke_width=3):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()
    dummy = Image.new("RGBA", (1, 1))
    draw  = ImageDraw.Draw(dummy)
    bbox  = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + stroke_width * 2 + 20
    h = bbox[3] - bbox[1] + stroke_width * 2 + 20
    img  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for dx in range(-stroke_width, stroke_width + 1):
        for dy in range(-stroke_width, stroke_width + 1):
            draw.text((stroke_width + 10 + dx, stroke_width + 10 + dy), text, font=font, fill=stroke_color)
    draw.text((stroke_width + 10, stroke_width + 10), text, font=font, fill=color)
    return np.array(img)


def make_number_overlay(number: int, day: int, duration: float):
    day_arr      = make_text_image("Day " + str(day), 60, "#FFC800")
    followers_arr = make_text_image("Followers " + str(number), 60, "white")
    day_clip      = ImageClip(day_arr).set_duration(duration).set_position(("center", 0.22), relative=True)
    followers_clip = ImageClip(followers_arr).set_duration(duration).set_position(("center", 0.29), relative=True)
    return [day_clip, followers_clip]


def crop_to_portrait(clip):
    target_ratio = REEL_W / REEL_H
    clip_ratio   = clip.w / clip.h
    if clip_ratio > target_ratio:
        new_w = int(clip.h * target_ratio)
        clip  = crop(clip, width=new_w, x_center=clip.w / 2)
    else:
        new_h = int(clip.w / target_ratio)
        clip  = crop(clip, height=new_h, y_center=clip.h / 2)
    return resize(clip, (REEL_W, REEL_H))


def build_video(data: dict, audio_path: str, day: int, followers: int, output_path: str = "reel.mp4") -> str:
    audio        = AudioFileClip(audio_path)
    duration     = audio.duration
    segments     = data["segments"]
    n            = len(segments)
    seg_duration = duration / n
    print(f"[INFO] Total duration: {duration:.1f}s, {n} segments x {seg_duration:.1f}s each")

    clips = []
    for i, seg in enumerate(segments):
        clip_path   = f"clip_{i}.mp4"
        pexels_path = fetch_pexels_video(seg["keywords"], clip_path)
        if pexels_path:
            try:
                raw = VideoFileClip(pexels_path)
                if raw.duration < seg_duration:
                    loops = int(seg_duration / raw.duration) + 1
                    raw   = concatenate_videoclips([raw] * loops)
                raw = raw.subclip(0, seg_duration)
                raw = crop_to_portrait(raw).without_audio()
            except Exception as e:
                print(f"[WARN] Failed to process clip {i}: {e}, using color fallback")
                raw = ColorClip(size=(REEL_W, REEL_H), color=(15, 15, 30)).set_duration(seg_duration)
        else:
            raw = ColorClip(size=(REEL_W, REEL_H), color=(15, 15, 30)).set_duration(seg_duration)
        overlay = ColorClip(size=(REEL_W, REEL_H), color=(0, 0, 0)).set_opacity(0.45).set_duration(seg_duration)
        clips.append(CompositeVideoClip([raw, overlay], size=(REEL_W, REEL_H)))

    base_video = concatenate_videoclips(clips, method="compose")
    overlays   = make_number_overlay(followers, day, duration)
    final      = CompositeVideoClip([base_video] + overlays, size=(REEL_W, REEL_H)).set_audio(audio).set_duration(duration)
    final.write_videofile(
        output_path, fps=30, codec="libx264", audio_codec="aac",
        temp_audiofile="temp_audio.m4a", remove_temp=True, logger=None,
        ffmpeg_params=[
            "-profile:v", "baseline", "-level", "3.0",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-b:v", "3500k", "-b:a", "128k", "-ar", "44100", "-ac", "2"
        ]
    )
    print(f"[INFO] Final video built -> {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 6. Post reel using instagrapi
# ─────────────────────────────────────────────────────────────────────────────
def post_reel_instagrapi(video_path: str, caption: str):
    cl    = get_ig_client()
    print("[INFO] Uploading reel...")
    media = cl.clip_upload(path=video_path, caption=caption)
    print(f"[INFO] Reel posted! https://www.instagram.com/reel/{media.code}/")
    return media


# ─────────────────────────────────────────────────────────────────────────────
# 7. Orchestrate
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import imageio_ffmpeg
    os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

    print("=" * 60)
    print("  Daily Reel Generator -- " + datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    print("=" * 60)

    followers  = get_follower_count()
    data       = generate_script_and_keywords(DAY_NUMBER, followers)
    audio_path = generate_voiceover(data["script"])
    video_path = build_video(data, audio_path, DAY_NUMBER, followers)

    caption = (
        "Day " + str(DAY_NUMBER) + " of posting facts about the number " + str(followers) + "\n\n"
        + "Today we have " + str(followers) + " followers and THIS number is fascinating!\n\n"
        + "Follow us and be part of this journey to 1,000,000\n\n"
        + "#numberfacts #didyouknow #educational #facts #math "
        + "#instagramreels #viral #mindblown #science"
    )

    post_reel_instagrapi(video_path, caption)
    print("\nDone! Reel posted successfully.")


if __name__ == "__main__":
    main()
