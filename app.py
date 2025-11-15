from __future__ import annotations

import json
import mimetypes
import os
import threading
import uuid
import random
import wave

import numpy as np
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from moviepy import AudioFileClip, CompositeAudioClip, VideoFileClip
from moviepy.audio.fx.all import speedx, volumex
from scipy.signal import find_peaks

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "uploads")
OUTPUT_FOLDER = os.path.join(APP_ROOT, "outputs")
STATIC_FOLDER = os.path.join(APP_ROOT, "static")
SFX_FILENAME = "sound_effect.wav"
SFX_PATH = os.path.join(STATIC_FOLDER, SFX_FILENAME)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)


def ensure_default_sfx():
    """Create a simple default sound effect if none exists."""
    if os.path.exists(SFX_PATH):
        return

    sample_rate = 44100
    duration = 0.4
    frequency = 880.0
    amplitude = 0.5
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    waveform = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    with wave.open(SFX_PATH, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        pcm = np.int16(waveform * 32767)
        wav_file.writeframes(pcm.tobytes())


ensure_default_sfx()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

video_registry: dict[str, dict[str, str]] = {}
export_jobs: dict[str, dict[str, str | float]] = {}
registry_lock = threading.Lock()
export_lock = threading.Lock()


def detect_peaks(mono_audio: np.ndarray, prominence: float, min_spacing: float, sample_rate: int) -> list[float]:
    """Detect peak times in the provided mono audio array."""
    abs_audio = np.abs(mono_audio)
    if abs_audio.size == 0:
        return []

    prominence_value = prominence * np.max(abs_audio)
    if prominence_value <= 0:
        return []

    peak_indices, _ = find_peaks(abs_audio, prominence=prominence_value)
    if peak_indices.size == 0:
        return []

    peak_times = np.array(peak_indices) / sample_rate
    peak_values = abs_audio[peak_indices]

    filtered: list[float] = []
    group: list[tuple[float, float]] = []

    for time_point, value in zip(peak_times, peak_values):
        if not group:
            group.append((time_point, value))
            continue

        if time_point - group[-1][0] <= min_spacing:
            group.append((time_point, value))
        else:
            best = max(group, key=lambda item: item[1])
            filtered.append(best[0])
            group = [(time_point, value)]

    if group:
        best = max(group, key=lambda item: item[1])
        filtered.append(best[0])

    return filtered


def load_audio_array(audio_clip: AudioFileClip, target_rate: int = 44100) -> tuple[np.ndarray, int]:
    """Return a mono audio array sampled at the desired rate."""
    sound_array = audio_clip.to_soundarray(fps=target_rate)
    mono = sound_array.mean(axis=1) if sound_array.ndim > 1 else sound_array
    return mono, target_rate


def get_video_entry(video_id: str) -> dict[str, str] | None:
    with registry_lock:
        return video_registry.get(video_id)


def register_video(video_id: str, path: str, original_name: str) -> None:
    with registry_lock:
        video_registry[video_id] = {"path": path, "name": original_name}


def build_export_job(video_id: str, settings: dict) -> str:
    job_id = uuid.uuid4().hex
    with export_lock:
        export_jobs[job_id] = {
            "status": "processing",
            "percent": 0,
            "video_id": video_id,
            "download_url": "",
            "error": "",
            "settings": json.dumps(settings),
        }
    return job_id


def update_export_job(job_id: str, **kwargs):
    with export_lock:
        if job_id in export_jobs:
            export_jobs[job_id].update(kwargs)


def process_export_job(job_id: str, video_path: str, settings: dict):
    base_sfx: AudioFileClip | None = None
    final_audio: CompositeAudioClip | None = None
    final_video: VideoFileClip | None = None
    video: VideoFileClip | None = None
    audio = None
    try:
        update_export_job(job_id, percent=10)
        video = VideoFileClip(video_path)
        audio = video.audio

        update_export_job(job_id, percent=25)
        mono_audio, sample_rate = load_audio_array(audio)
        peaks = detect_peaks(
            mono_audio,
            settings["prominence"],
            settings["spacing"],
            sample_rate,
        )

        update_export_job(job_id, percent=45)
        base_sfx = AudioFileClip(SFX_PATH)
        overlays = [audio]

        for time_point in peaks:
            clip = base_sfx
            if settings["pitch_enabled"] and settings["pitch_amount"] > 0:
                variation = settings["pitch_amount"]
                factor_min = max(0.4, 1.0 - variation)
                factor_max = 1.0 + variation
                factor = random.uniform(factor_min, factor_max)
                clip = clip.fx(speedx, factor)

            volume_multiplier = settings["volume"]
            if volume_multiplier != 1.0:
                clip = clip.fx(volumex, volume_multiplier)

            overlays.append(clip.set_start(time_point))

        update_export_job(job_id, percent=60)
        final_audio = CompositeAudioClip(overlays)
        final_video = video.set_audio(final_audio)

        filename = f"{uuid.uuid4().hex}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        update_export_job(job_id, percent=75)
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=video.fps,
            logger=None,
        )

        update_export_job(job_id, percent=100, status="done", download_url=f"/download/{filename}")
    except Exception as exc:  # pragma: no cover - defensive path
        update_export_job(job_id, status="error", error=str(exc))
    finally:
        for clip in (base_sfx, final_audio, final_video, video, audio):
            if clip is None:
                continue
            try:
                clip.close()
            except Exception:
                pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    video_file = request.files.get("video")
    if not video_file:
        return redirect(url_for("index"))

    original_name = video_file.filename or "video"
    extension = os.path.splitext(original_name)[1] or ".mp4"
    video_id = uuid.uuid4().hex
    filename = f"{video_id}{extension}"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(save_path)

    register_video(video_id, save_path, original_name)
    return redirect(url_for("editor", video_id=video_id))


@app.route("/editor/<video_id>")
def editor(video_id: str):
    entry = get_video_entry(video_id)
    if not entry:
        return redirect(url_for("index"))

    video_url = url_for("serve_video", video_id=video_id, ext=os.path.splitext(entry["path"])[1])
    return render_template(
        "editor.html",
        video_id=video_id,
        video_url=video_url,
        sfx_url=url_for("static", filename=SFX_FILENAME),
    )


@app.route("/media/<video_id><ext>")
def serve_video(video_id: str, ext: str):
    entry = get_video_entry(video_id)
    if not entry:
        return "Not found", 404
    mime_type = mimetypes.guess_type(entry["path"])[0] or "application/octet-stream"
    return send_file(entry["path"], mimetype=mime_type)


@app.route("/api/peaks/<video_id>", methods=["POST"])
def api_peaks(video_id: str):
    entry = get_video_entry(video_id)
    if not entry:
        return jsonify({"error": "Video not found"}), 404

    settings = request.get_json(force=True)
    prominence = float(settings.get("prominence", 0.2))
    spacing = float(settings.get("spacing", 0.3))

    video = VideoFileClip(entry["path"])
    audio = video.audio
    mono_audio, sample_rate = load_audio_array(audio)
    peak_times = detect_peaks(mono_audio, prominence, spacing, sample_rate)
    duration = float(video.duration)

    video.close()
    audio.close()

    return jsonify({"peaks": peak_times, "duration": duration})


@app.route("/api/export/<video_id>", methods=["POST"])
def api_export(video_id: str):
    entry = get_video_entry(video_id)
    if not entry:
        return jsonify({"error": "Video not found"}), 404

    payload = request.get_json(force=True)
    settings = {
        "prominence": float(payload.get("prominence", 0.2)),
        "spacing": float(payload.get("spacing", 0.3)),
        "volume": float(payload.get("volume", 1.0)),
        "pitch_enabled": bool(payload.get("pitch_enabled", False)),
        "pitch_amount": float(payload.get("pitch_amount", 0.0)),
        "reverb": payload.get("reverb", "None"),
    }

    job_id = build_export_job(video_id, settings)
    thread = threading.Thread(
        target=process_export_job,
        args=(job_id, entry["path"], settings),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/export/status/<job_id>")
def export_status(job_id: str):
    with export_lock:
        job = export_jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(job)


@app.route("/download/<path:path>")
def download_file(path: str):
    full_path = os.path.join(OUTPUT_FOLDER, path)
    if not os.path.exists(full_path):
        return "Not found", 404
    return send_file(full_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
