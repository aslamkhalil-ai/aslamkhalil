"""CineForge AI - Local Piper Text-to-Speech Engine."""

import os
import shutil
import subprocess
import logging
import wave

from pathlib import Path


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

VOICE_MODEL = BASE_DIR / "en_US-lessac-medium.onnx"
VOICE_CONFIG = BASE_DIR / "en_US-lessac-medium.onnx.json"

PIPER_EXECUTABLE = os.getenv("PIPER_PATH") or shutil.which("piper")


def find_piper():
    """Find the Piper executable."""

    if PIPER_EXECUTABLE:
        return PIPER_EXECUTABLE

    # Check common local project locations.
    candidates = [
        BASE_DIR / "piper.exe",
        BASE_DIR / "piper" / "piper.exe",
        BASE_DIR / ".venv" / "Scripts" / "piper.exe"
    ]

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        "Piper executable not found. Install Piper or set "
        "the PIPER_PATH environment variable."
    )


def validate_wav(output_path):
    """Verify that the generated audio is a valid WAV file."""

    if not output_path.is_file():
        raise RuntimeError("Piper did not create the narration file.")

    if output_path.stat().st_size < 1000:
        raise RuntimeError("Generated narration file is unexpectedly small.")

    try:
        with wave.open(str(output_path), "rb") as audio:
            frames = audio.getnframes()
            sample_rate = audio.getframerate()
            channels = audio.getnchannels()

            if frames <= 0 or sample_rate <= 0 or channels <= 0:
                raise RuntimeError("Generated WAV file contains no audio.")

            duration = frames / sample_rate

            logger.info(
                "Narration generated: %.2f seconds, %s Hz, %s channel(s)",
                duration,
                sample_rate,
                channels
            )

    except (wave.Error, EOFError) as exc:
        raise RuntimeError(
            f"Generated narration is not a valid WAV file: {exc}"
        ) from exc


def generate_voice(text, output_path):
    """Generate local narration using Piper TTS."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Narration text is empty or invalid.")

    if not VOICE_MODEL.is_file():
        raise FileNotFoundError(
            f"Piper voice model not found: {VOICE_MODEL}"
        )

    if not VOICE_CONFIG.is_file():
        raise FileNotFoundError(
            f"Piper voice configuration not found: {VOICE_CONFIG}"
        )

    piper = find_piper()

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove an older output so a failed run cannot reuse stale audio.
    if output_path.exists():
        output_path.unlink()

    command = [
        piper,
        "--model", str(VOICE_MODEL),
        "--config", str(VOICE_CONFIG),
        "--output_file", str(output_path)
    ]

    try:
        result = subprocess.run(
            command,
            input=text.strip(),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=300,
            cwd=str(BASE_DIR)
        )

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Piper narration generation timed out after 5 minutes."
        ) from exc

    except OSError as exc:
        raise RuntimeError(
            f"Unable to start Piper: {exc}"
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            "Piper voice generation failed:\n"
            f"{result.stderr[-3000:]}"
        )

    validate_wav(output_path)

    logger.info("Narration saved: %s", output_path)

    return str(output_path)