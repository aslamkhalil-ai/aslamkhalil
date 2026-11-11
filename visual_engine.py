
"""
CineForge AI - Cinematic Visual & Video Engine

Creates cinematic video clips from AI-generated scene images
using FFmpeg.

Features:
- Automatic FFmpeg detection
- Ken Burns camera movements
- Cinematic fades and image enhancement
- Consistent H.264 video encoding
- Scene clip assembly
- Optional audio and music mixing
- CPU-friendly rendering
"""

import os
import shutil
import subprocess
import logging

from pathlib import Path


logger = logging.getLogger(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

WIDTH = 1280
HEIGHT = 720
FPS = 24

VIDEO_CODEC = "libx264"
VIDEO_PRESET = "ultrafast"
VIDEO_CRF = "20"

AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"

DEFAULT_TIMEOUT = 600


# =========================================================
# FFMPEG DETECTION
# =========================================================

def _resolve_ffmpeg():
    """Find FFmpeg from environment, PATH or Windows locations."""

    configured_path = os.getenv("FFMPEG_PATH")

    if configured_path:
        candidate = Path(configured_path.strip('"'))

        if candidate.is_file():
            return str(candidate)

    executable = shutil.which("ffmpeg")

    if executable:
        return executable

    windows_candidates = [
        Path(
            os.getenv("LOCALAPPDATA", "")
        ) / "Microsoft" / "WinGet" / "Packages" /
        "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe" /
        "ffmpeg-9.0.2-full_build" / "bin" / "ffmpeg.exe",

        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
    ]

    for candidate in windows_candidates:
        if candidate.is_file():
            return str(candidate)

    return "ffmpeg"


FFMPEG_PATH = _resolve_ffmpeg()


def _check_ffmpeg():
    """Verify that FFmpeg is executable."""

    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr or "FFmpeg could not start."
            )

        return True

    except FileNotFoundError as error:
        raise FileNotFoundError(
            "FFmpeg was not found. Install FFmpeg or configure FFMPEG_PATH."
        ) from error

    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "FFmpeg did not respond to the version check."
        ) from error


# =========================================================
# FFMPEG EXECUTION
# =========================================================

def run_ffmpeg(command, timeout=DEFAULT_TIMEOUT):
    """Execute FFmpeg and return useful error information."""

    if not command:
        raise ValueError("FFmpeg command cannot be empty.")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            error_output = (
                result.stderr[-5000:]
                if result.stderr
                else "Unknown FFmpeg error."
            )

            logger.error(
                "FFmpeg execution failed:\n%s",
                error_output,
            )

            raise RuntimeError(
                f"FFmpeg operation failed:\n{error_output}"
            )

        return result

    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"FFmpeg rendering exceeded {timeout} seconds."
        ) from error

    except FileNotFoundError as error:
        raise RuntimeError(
            "FFmpeg executable could not be located."
        ) from error


# =========================================================
# FILE VALIDATION
# =========================================================

def validate_output(path):
    """Verify that an output file exists and is not empty."""

    path = Path(path)

    if not path.is_file():
        raise RuntimeError(
            f"Output file was not created: {path}"
        )

    if path.stat().st_size == 0:
        raise RuntimeError(
            f"Output file is empty: {path}"
        )

    return True


def _validate_input(path, description="Input file"):
    """Validate an input media file."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"{description} is empty: {path}"
        )

    return path.resolve()


def _validate_dimensions(width, height, fps):
    """Validate output dimensions and frame rate."""

    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError(
            "Video dimensions and FPS must be positive."
        )

    if width % 2 != 0 or height % 2 != 0:
        raise ValueError(
            "Video dimensions must be even numbers."
        )


# =========================================================
# CAMERA MOVEMENT
# =========================================================

def _get_camera_filter(
    motion,
    total_frames,
    width=WIDTH,
    height=HEIGHT,
    fps=FPS,
):
    """Build a Ken Burns camera movement filter."""

    supported_motions = {
        "zoom_in",
        "zoom_out",
        "pan_left",
        "pan_right",
        "pan_up",
        "pan_down",
        "static",
    }

    if motion not in supported_motions:
        raise ValueError(
            f"Unsupported camera movement: {motion}"
        )

    frame_range = max(total_frames - 1, 1)

    center_x = "iw/2-(iw/zoom/2)"
    center_y = "ih/2-(ih/zoom/2)"

    if motion == "zoom_in":
        zoom = "min(1.0+0.0005*on,1.12)"
        x = center_x
        y = center_y

    elif motion == "zoom_out":
        zoom = "max(1.0,1.12-0.0005*on)"
        x = center_x
        y = center_y

    elif motion == "pan_left":
        zoom = "1.10"
        x = f"(iw-iw/zoom)*on/{frame_range}"
        y = center_y

    elif motion == "pan_right":
        zoom = "1.10"
        x = f"(iw-iw/zoom)*(1-on/{frame_range})"
        y = center_y

    elif motion == "pan_up":
        zoom = "1.10"
        x = center_x
        y = f"(ih-ih/zoom)*on/{frame_range}"

    elif motion == "pan_down":
        zoom = "1.10"
        x = center_x
        y = f"(ih-ih/zoom)*(1-on/{frame_range})"

    else:
        zoom = "1.04"
        x = center_x
        y = center_y

    return (
        "zoompan="
        f"z='{zoom}':"
        f"x='{x}':"
        f"y='{y}':"
        f"d=1:"
        f"s={width}x{height}:"
        f"fps={fps}"
    )


# =========================================================
# CINEMATIC SCENE CLIP GENERATION
# =========================================================

def create_scene_clip(
    image_path,
    output_path,
    duration=5,
    width=1280,
    height=720,
    fps=24,
    motion="zoom_in",
):
    """
    Convert a generated scene image into a cinematic MP4.

    Applies:
    - Ken Burns camera movement
    - Fade-in and fade-out
    - Contrast and saturation enhancement
    - Controlled sharpening
    - H.264 encoding
    """

    _check_ffmpeg()

    image_path = _validate_input(
        image_path,
        "Scene image",
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        duration = float(duration)

    except (TypeError, ValueError) as error:
        raise ValueError(
            "Invalid scene duration."
        ) from error

    if duration <= 0:
        raise ValueError(
            "Scene duration must be greater than zero."
        )

    _validate_dimensions(width, height, fps)

    total_frames = max(
        1,
        round(duration * fps),
    )

    fade_duration = min(
        0.65,
        duration / 4,
    )

    camera_filter = _get_camera_filter(
        motion,
        total_frames,
        width=width,
        height=height,
        fps=fps,
    )

    filter_complex = (
        f"scale={width * 2}:{height * 2}:"
        "force_original_aspect_ratio=increase,"
        f"crop={width * 2}:{height * 2},"
        f"{camera_filter},"
        f"fade=t=in:st=0:d={fade_duration},"
        f"fade=t=out:st={max(0, duration - fade_duration)}:"
        f"d={fade_duration},"
        "eq=contrast=1.05:saturation=1.08:brightness=0.005,"
        "unsharp=5:5:0.45:5:5:0.0,"
        "vignette=PI/5,"
        "format=yuv420p"
    )

    command = [
        FFMPEG_PATH,
        "-y",
        "-hide_banner",
        "-loglevel", "error",

        "-loop", "1",
        "-framerate", str(fps),
        "-i", str(image_path),

        "-vf", filter_complex,
        "-frames:v", str(total_frames),
        "-an",

        "-c:v", VIDEO_CODEC,
        "-preset", VIDEO_PRESET,
        "-crf", VIDEO_CRF,
        "-threads", "2",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",

        str(output_path),
    ]

    run_ffmpeg(
        command,
        timeout=300,
    )

    validate_output(output_path)

    logger.info(
        "Scene clip created: %s | Motion: %s",
        output_path,
        motion,
    )

    return str(output_path)


# =========================================================
# SCENE CLIP ASSEMBLY
# =========================================================

def combine_scene_clips(
    clip_paths,
    output_path,
):
    """
    Combine scene clips into a single MP4.

    Uses FFmpeg concat demuxer and re-encodes
    the combined video for consistent output.
    """

    _check_ffmpeg()

    if not clip_paths:
        raise ValueError(
            "No scene clips provided."
        )

    clips = [
        _validate_input(path, "Scene clip")
        for path in clip_paths
    ]

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    list_file = (
        output_path.parent / "scene_list.txt"
    )

    try:
        with open(
            list_file,
            "w",
            encoding="utf-8",
        ) as file:

            for clip in clips:
                safe_path = str(clip).replace("\\", "/")
                safe_path = safe_path.replace("'", "'\\''")

                file.write(
                    f"file '{safe_path}'\n"
                )

        command = [
            FFMPEG_PATH,
            "-y",
            "-hide_banner",
            "-loglevel", "error",

            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),

            "-an",

            "-c:v", VIDEO_CODEC,
            "-preset", VIDEO_PRESET,
            "-crf", VIDEO_CRF,
            "-r", str(FPS),
            "-threads", "2",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",

            str(output_path),
        ]

        run_ffmpeg(
            command,
            timeout=600,
        )

        validate_output(output_path)

        logger.info(
            "Combined %s scene clips into %s",
            len(clips),
            output_path,
        )

        return str(output_path)

    finally:
        if list_file.exists():
            list_file.unlink()


# =========================================================
# BACKGROUND MUSIC
# =========================================================

def add_background_music(
    video_path,
    music_path,
    output_path,
    music_volume=0.18,
):
    """
    Mix background music with a video that already has audio.

    The input video must contain an audio stream.
    """

    _check_ffmpeg()

    video_path = _validate_input(
        video_path,
        "Video",
    )

    music_path = _validate_input(
        music_path,
        "Music",
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not 0 <= music_volume <= 1:
        raise ValueError(
            "Music volume must be between 0 and 1."
        )

    filter_complex = (
        f"[1:a]volume={music_volume}[bg];"
        "[0:a][bg]amix=inputs=2:"
        "duration=first:"
        "dropout_transition=2[a]"
    )

    command = [
        FFMPEG_PATH,
        "-y",
        "-hide_banner",
        "-loglevel", "error",

        "-i", str(video_path),
        "-stream_loop", "-1",
        "-i", str(music_path),

        "-filter_complex", filter_complex,

        "-map", "0:v:0",
        "-map", "[a]",

        "-c:v", "copy",
        "-c:a", AUDIO_CODEC,
        "-b:a", AUDIO_BITRATE,
        "-shortest",
        "-movflags", "+faststart",

        str(output_path),
    ]

    run_ffmpeg(command)

    validate_output(output_path)

    return str(output_path)


# =========================================================
# FINAL VIDEO AUDIO MIXING
# =========================================================

def render_final_video(
    video_path,
    voice_path,
    music_path,
    output_path,
    music_volume=0.18,
):
    """
    Combine scene video, narration and optional music.

    Narration remains foreground audio.
    Music is mixed at a lower volume.
    """

    _check_ffmpeg()

    video_path = _validate_input(
        video_path,
        "Scene video",
    )

    voice_path = _validate_input(
        voice_path,
        "Narration",
    )

    if music_path is not None:
        music_path = _validate_input(
            music_path,
            "Background music",
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not 0 <= music_volume <= 1:
        raise ValueError(
            "Music volume must be between 0 and 1."
        )

    command = [
        FFMPEG_PATH,
        "-y",
        "-hide_banner",
        "-loglevel", "error",

        "-i", str(video_path),
        "-i", str(voice_path),
    ]

    if music_path:
        command.extend([
            "-stream_loop", "-1",
            "-i", str(music_path),
        ])

        filter_complex = (
            "[1:a]volume=1.0[voice];"
            f"[2:a]volume={music_volume}[music];"
            "[voice][music]amix=inputs=2:"
            "duration=first:"
            "dropout_transition=2:"
            "normalize=0[audio]"
        )

    else:
        filter_complex = (
            "[1:a]volume=1.0[audio]"
        )

    command.extend([
        "-filter_complex", filter_complex,

        "-map", "0:v:0",
        "-map", "[audio]",

        "-c:v", "copy",
        "-c:a", AUDIO_CODEC,
        "-b:a", AUDIO_BITRATE,

        "-shortest",
        "-movflags", "+faststart",

        str(output_path),
    ])

    run_ffmpeg(
        command,
        timeout=600,
    )

    validate_output(output_path)

    logger.info(
        "Final video audio mix created: %s",
        output_path,
    )

    return str(output_path)


# =========================================================
# ENGINE HEALTH CHECK
# =========================================================

def check_ffmpeg():
    """Public FFmpeg availability check."""

    return _check_ffmpeg()
