
"""
CineForge AI - Cinematic Final Composition Engine

Combines scene video, narration, background music
and subtitles into the final production video.
"""

import subprocess
from pathlib import Path

from engines.visual_engine import FFMPEG_PATH


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

VIDEO_CODEC = "libx264"
VIDEO_PRESET = "ultrafast"
VIDEO_CRF = "20"

AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"

THREADS = "2"

NARRATION_VOLUME = 1.0
MUSIC_VOLUME = 0.12


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def _validate_file(path, label):

    file_path = Path(path).resolve()

    if not file_path.is_file():
        raise FileNotFoundError(
            f"{label} file not found: {file_path}"
        )

    if file_path.stat().st_size == 0:
        raise ValueError(
            f"{label} file is empty: {file_path}"
        )

    return file_path


def _validate_ffmpeg():

    if not FFMPEG_PATH:
        raise RuntimeError(
            "FFmpeg executable was not found."
        )

    ffmpeg = Path(FFMPEG_PATH)

    if not ffmpeg.is_file():
        raise FileNotFoundError(
            f"FFmpeg executable not found: {ffmpeg}"
        )

    return str(ffmpeg)


# ---------------------------------------------------------
# SUBTITLE PATH ESCAPING
# ---------------------------------------------------------

def _escape_subtitle_path(path):
    """
    Escapes a Windows file path for FFmpeg subtitles filter.
    """

    value = str(Path(path).resolve())

    # FFmpeg filter paths work more reliably with forward slashes.
    value = value.replace("\\", "/")

    # Escape special characters used by FFmpeg filter syntax.
    value = value.replace(":", r"\:")
    value = value.replace("'", r"\'")
    value = value.replace(",", r"\,")
    value = value.replace("[", r"\[")
    value = value.replace("]", r"\]")

    return value


# ---------------------------------------------------------
# FINAL VIDEO COMPOSITION
# ---------------------------------------------------------

def create_final_video(
    video_path,
    narration_path,
    subtitle_path,
    output_path,
    music_path=None
):
    """
    Creates the final cinematic video.

    Inputs:
        video_path: Combined scene video.
        narration_path: Piper-generated narration WAV.
        subtitle_path: SRT subtitle file.
        output_path: Final MP4 destination.
        music_path: Optional local background music.

    Returns:
        Absolute path to final MP4.
    """

    ffmpeg = _validate_ffmpeg()

    video = _validate_file(
        video_path,
        "Scene video"
    )

    narration = _validate_file(
        narration_path,
        "Narration"
    )

    subtitles = _validate_file(
        subtitle_path,
        "Subtitle"
    )

    output = Path(output_path).resolve()

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if output.exists():
        output.unlink()

    music = None

    if music_path:

        music_candidate = Path(music_path).resolve()

        if music_candidate.is_file() and music_candidate.stat().st_size > 0:
            music = music_candidate

    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video),
        "-i", str(narration),
    ]

    if music:
        command.extend([
            "-stream_loop", "-1",
            "-i", str(music)
        ])

    # -----------------------------------------------------
    # VIDEO FILTER
    # -----------------------------------------------------

    subtitle_file = _escape_subtitle_path(subtitles)

    subtitle_style = (
        "FontName=Arial,"
        "FontSize=19,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00101010,"
        "BackColour=&H80000000,"
        "BorderStyle=1,"
        "Outline=2,"
        "Shadow=1,"
        "Alignment=2,"
        "MarginL=110,"
        "MarginR=110,"
        "MarginV=48,"
        "WrapStyle=2"
    )

    video_filter = (
        f"subtitles=filename='{subtitle_file}':"
        f"force_style='{subtitle_style}'"
    )

    command.extend([
        "-vf", video_filter,
        "-map", "0:v:0",
    ])

    # -----------------------------------------------------
    # AUDIO MIXING
    # -----------------------------------------------------

    if music:

        audio_filter = (
            f"[1:a]volume={NARRATION_VOLUME},"
            "aresample=async=1:first_pts=0[narration];"
            f"[2:a]volume={MUSIC_VOLUME},"
            "aresample=async=1:first_pts=0[music];"
            "[narration][music]"
            "amix=inputs=2:duration=first:"
            "dropout_transition=2:normalize=0[audio]"
        )

        command.extend([
            "-filter_complex", audio_filter,
            "-map", "[audio]",
        ])

    else:

        command.extend([
            "-map", "1:a:0",
            "-af", "aresample=async=1:first_pts=0",
        ])

    # -----------------------------------------------------
    # OUTPUT SETTINGS
    # -----------------------------------------------------

    command.extend([
        "-c:v", VIDEO_CODEC,
        "-preset", VIDEO_PRESET,
        "-crf", VIDEO_CRF,
        "-pix_fmt", "yuv420p",
        "-r", "24",
        "-c:a", AUDIO_CODEC,
        "-b:a", AUDIO_BITRATE,
        "-ar", "44100",
        "-threads", THREADS,
        "-shortest",
        "-movflags", "+faststart",
        str(output),
    ])

    # -----------------------------------------------------
    # EXECUTION
    # -----------------------------------------------------

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=900,
            check=False
        )

        if result.returncode != 0:

            error_message = (
                result.stderr.strip()
                or "Unknown FFmpeg rendering error."
            )

            raise RuntimeError(
                f"Final video rendering failed:\n{error_message[-4000:]}"
            )

        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(
                "FFmpeg completed but the final video is missing or empty."
            )

        return str(output)

    except subprocess.TimeoutExpired as exc:

        raise TimeoutError(
            "Final video rendering exceeded the 15-minute limit."
        ) from exc

    except Exception:

        if output.exists() and output.stat().st_size == 0:
            output.unlink(missing_ok=True)

        raise
