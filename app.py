# Flask App with Improved GUI, Sliders, and Progress Bar
# ------------------------------------------------------
# This document includes:
#  - Updated app.py (Flask backend)
#  - index.html with nicer GUI, sliders, and drag/drop
#  - result.html with video player and download button
#  - Simple progress polling endpoint
# ------------------------------------------------------

"""
NOTE: Copy the HTML parts into templates/index.html and templates/result.html.
This file shows all code together for convenience.
"""

# ========================= app.py =============================
from flask import Flask, render_template, request, send_file, jsonify
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
import numpy as np
from scipy.signal import find_peaks
import uuid
import os
import threading

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
SFX_PATH = "static/sound_effect.wav"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global progress variable
progress = {"status": "idle", "percent": 0}

def update_progress(p):
    progress["percent"] = p

# ---------------------------------------------------------
# Smart Peak Detection (with non-maximum suppression)
# ---------------------------------------------------------
def detect_peaks(mono_audio, prominence_factor, min_spacing_sec, sample_rate=44100):
    abs_audio = np.abs(mono_audio)

    # Initial detection
    peak_indices, _ = find_peaks(
        abs_audio,
        prominence=prominence_factor * np.max(abs_audio)
    )

    peak_times = np.array(peak_indices) / sample_rate
    peak_values = abs_audio[peak_indices]

    # Grouping for non-maximum suppression
    peaks = list(zip(peak_times, peak_values))
    filtered = []
    group = []

    for t, val in peaks:
        if not group:
            group.append((t, val))
        else:
            if t - group[-1][0] <= min_spacing_sec:
                group.append((t, val))
            else:
                best = max(group, key=lambda p: p[1])
                filtered.append(best[0])
                group = [(t, val)]

    if group:
        best = max(group, key=lambda p: p[1])
        filtered.append(best[0])

    return filtered

# ---------------------------------------------------------
# VIDEO PROCESSING THREAD
# ---------------------------------------------------------
def process_video_async(video_path, sfx_path, prominence_factor, min_spacing):
    update_progress(5)
    progress["status"] = "processing"

    video = VideoFileClip(video_path)
    audio = video.audio

    update_progress(20)
    audio_array = audio.to_soundarray(fps=44100)
    sample_rate = 44100
    mono_audio = audio_array.mean(axis=1)

    update_progress(40)
    peaks = detect_peaks(mono_audio, prominence_factor, min_spacing, sample_rate)

    sfx = AudioFileClip(sfx_path)

    update_progress(60)
    layers = [audio]
    for t in peaks:
        layers.append(sfx.with_start(t))

    final_audio = CompositeAudioClip(layers)
    final_video = video.with_audio(final_audio)

    update_progress(80)
    output_filename = f"{uuid.uuid4()}.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=video.fps
    )

    final_video.close()
    video.close()
    audio.close()
    sfx.close()

    update_progress(100)
    progress["status"] = output_path.replace("\\", "/")

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    global progress

    progress = {"status": "processing", "percent": 0}

    file = request.files["video"]
    prominence_factor = float(request.form["sensitivity"])
    min_spacing = float(request.form["spacing"])  # seconds

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    threading.Thread(target=process_video_async, args=(input_path, SFX_PATH, prominence_factor, min_spacing)).start()

    return jsonify({"message": "processing started"})

@app.route("/progress")
def get_progress():
    return jsonify(progress)

@app.route("/result")
def result_page():
    return render_template("result.html")

@app.route("/video/<path:filename>")
def serve_video(filename):
    return send_file(filename, mimetype="video/mp4")

@app.route("/download/<path:filename>")
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
