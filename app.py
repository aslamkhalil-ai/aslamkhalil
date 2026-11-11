"""
CineForge AI
Local-first AI video production platform.

Pipeline:
Creative Brief -> AI Script -> AI Visuals -> Voice -> Subtitles
-> Scene Composition -> Music -> Final MP4

All AI processing runs locally.
"""

import json
import logging
import os
import wave

from pathlib import Path

import requests

from dotenv import load_dotenv

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from database import (
    init_db,
    create_project,
    get_projects,
    get_project,
    update_project,
)

from engines.story_engine import generate_story
from engines.voice_engine import generate_voice
from engines.subtitle_engine import generate_subtitles

from engines.visual_engine import (
    create_scene_clip,
    combine_scene_clips,
)

from engines.storyboard_engine import (
    create_storyboard_image,
    select_camera_motion,
)

from engines.composition_engine import create_final_video
from engines.music_manager import get_music_for_style
from dotenv import load_dotenv



# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2",
)

MIN_VIDEO_DURATION = 15
MAX_VIDEO_DURATION = 180

app = Flask(__name__)

app.config.update(
    SECRET_KEY=os.getenv(
        "FLASK_SECRET_KEY",
    ) or os.urandom(32),

    MAX_CONTENT_LENGTH=16 * 1024 * 1024,

    SEND_FILE_MAX_AGE_DEFAULT=0,
)

app.logger.setLevel(logging.INFO)

# Initialize the local SQLite database.
init_db()


# =========================================================
# COMMON HELPERS
# =========================================================

def get_project_directory(project_id):
    """Create and return a project's output directory."""

    project_dir = BASE_DIR / "projects" / str(int(project_id))

    project_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return project_dir


def get_safe_project_file(relative_path):
    """Resolve a database-stored path inside the application folder."""

    if not relative_path:
        raise ValueError("Project file path is missing.")

    relative_path = Path(relative_path)

    if relative_path.is_absolute():
        raise ValueError("Absolute file paths are not allowed.")

    file_path = (BASE_DIR / relative_path).resolve()

    try:
        file_path.relative_to(BASE_DIR)

    except ValueError as exc:
        raise ValueError(
            "Invalid project file path."
        ) from exc

    return file_path


def validate_generated_file(file_path, label):
    """Check that an output file exists and is not empty."""

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"{label} was not generated: {file_path}"
        )

    if file_path.stat().st_size == 0:
        raise RuntimeError(
            f"{label} is empty: {file_path}"
        )

    return file_path


def get_project_or_404(project_id):
    """Retrieve a project from the database."""

    return get_project(project_id)


def update_project_error(project_id, message):
    """Save a generation error without hiding the original exception."""

    try:
        update_project(
            project_id,
            status="Generation failed",
            error=str(message)[:2000],
        )

    except Exception:
        app.logger.exception(
            "Could not save error for project %s",
            project_id,
        )


def _audio_duration_seconds(audio_path):
    """Return WAV duration in seconds."""

    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    with wave.open(str(audio_path), "rb") as audio:

        frames = audio.getnframes()
        sample_rate = audio.getframerate()

        if sample_rate <= 0:
            raise ValueError(
                "Narration has an invalid sample rate."
            )

        duration = frames / float(sample_rate)

        if duration <= 0:
            raise ValueError(
                "Narration audio has no duration."
            )

        return duration


def _scene_durations(scenes, total_seconds):
    """
    Distribute narration duration across scenes.

    Uses scene duration weights where available.
    """

    if not scenes:
        raise ValueError(
            "Cannot calculate scene durations without scenes."
        )

    if total_seconds <= 0:
        raise ValueError(
            "Narration duration must be positive."
        )

    weights = []

    for scene in scenes:

        if isinstance(scene, dict):
            raw_duration = scene.get(
                "duration_seconds",
                scene.get("duration", 1),
            )
        else:
            raw_duration = 1

        try:
            weight = float(raw_duration)

        except (TypeError, ValueError):
            weight = 1.0

        weights.append(max(0.1, weight))

    scene_count = len(weights)

    # At least one second for each scene.
    target_duration = max(
        scene_count,
        round(total_seconds),
    )

    remaining = target_duration - scene_count

    total_weight = sum(weights)

    exact_values = [
        remaining * weight / total_weight
        for weight in weights
    ]

    integer_values = [
        int(value)
        for value in exact_values
    ]

    leftover = remaining - sum(integer_values)

    remainder_order = sorted(
        range(scene_count),
        key=lambda i: exact_values[i] - integer_values[i],
        reverse=True,
    )

    for index in remainder_order[:leftover]:
        integer_values[index] += 1

    return [
        1 + value
        for value in integer_values
    ]


def _load_project_scenes(project):
    """Parse scene JSON from the database."""

    raw_scenes = project.get("scenes") or "[]"

    if isinstance(raw_scenes, str):

        try:
            scenes = json.loads(raw_scenes)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid scene JSON. Regenerate the story plan."
            ) from exc

    elif isinstance(raw_scenes, list):
        scenes = raw_scenes

    else:
        raise ValueError(
            "Scene data has an unsupported format."
        )

    if not isinstance(scenes, list) or not scenes:
        raise ValueError(
            "No scenes found. Generate the AI story first."
        )

    for index, scene in enumerate(scenes):

        if not isinstance(scene, dict):
            raise ValueError(
                f"Scene {index + 1} has invalid data."
            )

    return scenes


def _normalize_story_scenes(scenes):
    """Normalize story-engine scenes into a Python list."""

    if isinstance(scenes, str):

        try:
            scenes = json.loads(scenes)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Story engine returned invalid scene JSON."
            ) from exc

    if not isinstance(scenes, list) or not scenes:
        raise ValueError(
            "Story engine did not return a valid scene list."
        )

    for index, scene in enumerate(scenes):

        if not isinstance(scene, dict):
            raise ValueError(
                f"Invalid scene data at position {index + 1}."
            )

    return scenes


def _redirect_to_project(project_id):
    return redirect(
        url_for(
            "project_detail",
            project_id=project_id,
        )
    )


def _get_existing_or_default_path(project, key, default):
    """Return a safe path for an existing or new project output."""

    relative_path = project.get(key) or default

    return get_safe_project_file(relative_path)


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/")
def dashboard():

    projects = get_projects()

    return render_template(
        "dashboard.html",
        projects=projects,
        active="dashboard",
    )


# =========================================================
# CREATE PROJECT
# =========================================================

@app.route(
    "/projects/new",
    methods=["GET", "POST"],
)
def new_project():

    if request.method == "GET":

        return render_template(
            "new_project.html",
            active="new",
        )

    title = request.form.get(
        "title",
        "",
    ).strip()

    idea = request.form.get(
        "idea",
        "",
    ).strip()

    genre = request.form.get(
        "genre",
        "Mystery",
    ).strip()

    duration_value = request.form.get(
        "duration",
        "60",
    ).strip()

    if not title or not idea:

        flash(
            "Please provide a project title and creative idea.",
            "error",
        )

        return render_template(
            "new_project.html",
            active="new",
        )

    if len(title) > 150 or len(idea) > 5000:

        flash(
            "Title or creative idea is too long.",
            "error",
        )

        return render_template(
            "new_project.html",
            active="new",
        )

    try:

        duration = int(duration_value)

        if not MIN_VIDEO_DURATION <= duration <= MAX_VIDEO_DURATION:
            raise ValueError

    except (ValueError, TypeError):

        flash(
            f"Duration must be between "
            f"{MIN_VIDEO_DURATION} and {MAX_VIDEO_DURATION} seconds.",
            "error",
        )

        return render_template(
            "new_project.html",
            active="new",
        )

    project_id = create_project(
        title,
        idea,
        genre,
        str(duration),
    )

    try:

        result = generate_story(
            idea,
            genre,
            duration,
        )

        if not isinstance(result, dict):
            raise ValueError(
                "Story engine returned an invalid response."
            )

        script = result.get("script", "").strip()

        scenes = _normalize_story_scenes(
            result.get("scenes")
        )

        if not script:
            raise ValueError(
                "AI did not generate a valid script."
            )

        update_data = {
            "status": "Script ready",
            "script": script,
            "scenes": json.dumps(
                scenes,
                ensure_ascii=False,
            ),
            "error": "",
        }

        # Use the generated title only if one is supplied.
        generated_title = result.get("title")

        if generated_title and len(generated_title) <= 150:
            update_data["title"] = generated_title

        update_project(
            project_id,
            **update_data,
        )

        flash(
            "Project created and AI script generated successfully!",
            "success",
        )

    except Exception as exc:

        app.logger.exception(
            "Story generation failed for project %s",
            project_id,
        )

        update_project_error(
            project_id,
            exc,
        )

        flash(
            "Project saved, but story generation failed. "
            "Check Ollama and try again.",
            "warning",
        )

    return _redirect_to_project(project_id)


# =========================================================
# PROJECT DETAILS
# =========================================================

@app.route("/projects/<int:project_id>")
def project_detail(project_id):

    project = get_project_or_404(project_id)

    if not project:
        return "Project not found", 404

    return render_template(
        "project_detail.html",
        project=project,
        active="dashboard",
    )


# =========================================================
# SYSTEM HEALTH
# =========================================================

@app.route("/api/health")
def health():

    result = {
        "application": "CineForge AI",
        "flask": "online",
        "ollama": "offline",
        "model": OLLAMA_MODEL,
    }

    try:

        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=3,
        )

        response.raise_for_status()

        models = response.json().get(
            "models",
            [],
        )

        result["ollama"] = "online"

        result["models"] = [
            model.get("name")
            for model in models
        ]

        return jsonify(result)

    except requests.RequestException as exc:

        result["detail"] = str(exc)

        return jsonify(result), 503

    except (ValueError, TypeError) as exc:

        result["ollama"] = "error"
        result["detail"] = str(exc)

        return jsonify(result), 503


# =========================================================
# GENERATE VOICE
# =========================================================

@app.route(
    "/projects/<int:project_id>/generate-voice",
    methods=["POST"],
)
def generate_project_voice(project_id):

    project = get_project_or_404(project_id)

    if not project:
        return "Project not found", 404

    try:

        script = project.get("script", "")

        if not script:
            raise ValueError(
                "Generate the story script first."
            )

        project_dir = get_project_directory(project_id)

        output_path = project_dir / "narration.wav"

        update_project(
            project_id,
            status="Generating narration",
            error="",
        )

        generate_voice(
            script,
            str(output_path),
        )

        validate_generated_file(
            output_path,
            "Narration",
        )

        update_project(
            project_id,
            voice_path=f"projects/{project_id}/narration.wav",
            status="Voice ready",
            error="",
        )

        flash(
            "Voice narration generated successfully!",
            "success",
        )

    except Exception as exc:

        app.logger.exception(
            "Voice generation failed for project %s",
            project_id,
        )

        update_project_error(
            project_id,
            exc,
        )

        flash(
            f"Voice generation failed: {exc}",
            "error",
        )

    return _redirect_to_project(project_id)


# =========================================================
# PLAY VOICE
# =========================================================

@app.route("/projects/<int:project_id>/voice")
def play_project_voice(project_id):

    project = get_project_or_404(project_id)

    if not project or not project.get("voice_path"):
        return "Voice narration not found", 404

    try:

        audio_path = get_safe_project_file(
            project["voice_path"]
        )

        validate_generated_file(
            audio_path,
            "Narration",
        )

        return send_file(
            audio_path,
            mimetype="audio/wav",
            conditional=True,
        )

    except ValueError:
        return "Invalid audio file path", 400

    except FileNotFoundError:
        return "Audio file not found", 404


# =========================================================
# GENERATE SUBTITLES
# =========================================================

@app.route(
    "/projects/<int:project_id>/generate-subtitles",
    methods=["POST"],
)
def generate_project_subtitles(project_id):

    project = get_project_or_404(project_id)

    if not project:
        return "Project not found", 404

    try:

        if not project.get("voice_path"):
            raise ValueError(
                "Generate voice narration before subtitles."
            )

        audio_path = get_safe_project_file(
            project["voice_path"]
        )

        validate_generated_file(
            audio_path,
            "Narration",
        )

        project_dir = get_project_directory(project_id)

        subtitle_path = project_dir / "subtitles.srt"

        update_project(
            project_id,
            status="Generating subtitles",
            error="",
        )

        generate_subtitles(
            str(audio_path),
            str(subtitle_path),
        )

        validate_generated_file(
            subtitle_path,
            "Subtitles",
        )

        update_project(
            project_id,
            subtitle_path=f"projects/{project_id}/subtitles.srt",
            status="Subtitles ready",
            error="",
        )

        flash(
            "Synchronized subtitles generated successfully!",
            "success",
        )

    except Exception as exc:

        app.logger.exception(
            "Subtitle generation failed for project %s",
            project_id,
        )

        update_project_error(
            project_id,
            exc,
        )

        flash(
            f"Subtitle generation failed: {exc}",
            "error",
        )

    return _redirect_to_project(project_id)


# =========================================================
# DOWNLOAD SUBTITLES
# =========================================================

@app.route("/projects/<int:project_id>/subtitles")
def download_project_subtitles(project_id):

    project = get_project_or_404(project_id)

    if not project or not project.get("subtitle_path"):
        return "Subtitle file not found", 404

    try:

        subtitle_path = get_safe_project_file(
            project["subtitle_path"]
        )

        validate_generated_file(
            subtitle_path,
            "Subtitle",
        )

        return send_file(
            subtitle_path,
            mimetype="text/plain",
            as_attachment=True,
            download_name="subtitles.srt",
        )

    except ValueError:
        return "Invalid subtitle file path", 400

    except FileNotFoundError:
        return "Subtitle file not found", 404


# =========================================================
# GENERATE COMPLETE VIDEO
# =========================================================

@app.route(
    "/projects/<int:project_id>/generate-final-video",
    methods=["POST"],
)
def generate_project_final_video(project_id):

    project = get_project_or_404(project_id)

    if not project:
        return "Project not found", 404

    try:

        app.logger.info(
            "Starting final production for project %s",
            project_id,
        )

        script = project.get("script", "")

        if not script:
            raise ValueError(
                "Generate the story script before video production."
            )

        project_dir = get_project_directory(project_id)

        visuals_dir = project_dir / "scene_visuals"
        clips_dir = project_dir / "scene_clips"

        visuals_dir.mkdir(parents=True, exist_ok=True)
        clips_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------
        # 1. Narration
        # -------------------------------------------------

        voice_relative_path = (
            project.get("voice_path")
            or f"projects/{project_id}/narration.wav"
        )

        voice_file = get_safe_project_file(
            voice_relative_path
        )

        app.logger.info(
    "Generating fresh narration from the current script..."
)
        generate_voice(
    script,
    str(voice_file),
)

        validate_generated_file(
    voice_file,
    "Narration",
)

        update_project(
            project_id,
            voice_path=voice_relative_path,
        )

        # -------------------------------------------------
        # 2. Subtitles
        # -------------------------------------------------

        subtitle_relative_path = (
            project.get("subtitle_path")
            or f"projects/{project_id}/subtitles.srt"
        )

        subtitle_file = get_safe_project_file(
            subtitle_relative_path
        )

        app.logger.info(
    "Generating fresh subtitles from the current narration..."
)

        generate_subtitles(
    str(voice_file),
    str(subtitle_file),
)

        validate_generated_file(
    subtitle_file,
    "Subtitles",
)

        update_project(
            project_id,
            subtitle_path=subtitle_relative_path,
        )

        # -------------------------------------------------
        # 3. Scene planning
        # -------------------------------------------------

        scenes = _load_project_scenes(project)

        narration_duration = _audio_duration_seconds(
            voice_file
        )

        scene_durations = _scene_durations(
            scenes,
            narration_duration,
        )

        clip_paths = []

        # -------------------------------------------------
        # 4. AI artwork and scene clips
        # -------------------------------------------------

        for index, scene in enumerate(scenes):

            app.logger.info(
                "Generating scene %s/%s",
                index + 1,
                len(scenes),
            )

            image_path = (
                visuals_dir / f"scene_{index + 1:02d}.jpg"
            )

            clip_path = (
                clips_dir / f"scene_{index + 1:02d}.mp4"
            )

            create_storyboard_image(
                scene,
                image_path,
                project.get("title", "CineForge AI"),
                index,
                len(scenes),
            )

            validate_generated_file(
                image_path,
                f"Scene {index + 1} image",
            )

            motion = select_camera_motion(
                scene,
                index,
            )

            create_scene_clip(
                str(image_path),
                str(clip_path),
                duration=scene_durations[index],
                motion=motion,
            )

            validate_generated_file(
                clip_path,
                f"Scene {index + 1} video",
            )

            clip_paths.append(
                str(clip_path)
            )

        # -------------------------------------------------
        # 5. Combine scenes
        # -------------------------------------------------

        visual_relative_path = (
            f"projects/{project_id}/scene_video.mp4"
        )

        visual_file = get_safe_project_file(
            visual_relative_path
        )

        app.logger.info(
            "Combining scene clips..."
        )

        combine_scene_clips(
            clip_paths,
            str(visual_file),
        )

        validate_generated_file(
            visual_file,
            "Combined scene video",
        )

        # -------------------------------------------------
        # 6. Background music
        # -------------------------------------------------

        music_file = None

        try:

            music_file = get_music_for_style(
                project.get("genre", "ambient")
            )

        except (FileNotFoundError, OSError) as exc:

            app.logger.warning(
                "Background music unavailable: %s",
                exc,
            )

        # -------------------------------------------------
        # 7. Final composition
        # -------------------------------------------------

        final_relative_path = (
            f"projects/{project_id}/final_video.mp4"
        )

        final_file = get_safe_project_file(
            final_relative_path
        )

        app.logger.info(
            "Rendering final MP4..."
        )

        create_final_video(
            str(visual_file),
            str(voice_file),
            str(subtitle_file),
            str(final_file),
            music_path=music_file,
        )

        validate_generated_file(
            final_file,
            "Final video",
        )

        # -------------------------------------------------
        # 8. Save production status
        # -------------------------------------------------

        update_project(
            project_id,
            visual_video_path=visual_relative_path,
            final_video_path=final_relative_path,
            voice_path=voice_relative_path,
            subtitle_path=subtitle_relative_path,
            status="Video ready",
            error="",
        )

        app.logger.info(
            "Production completed for project %s",
            project_id,
        )

        flash(
            "Final video rendered successfully. Your MP4 is ready!",
            "success",
        )

    except Exception as exc:

        app.logger.exception(
            "Final video production failed for project %s",
            project_id,
        )

        update_project_error(
            project_id,
            exc,
        )

        flash(
            f"Final video rendering failed: {exc}",
            "error",
        )

    return _redirect_to_project(project_id)


# =========================================================
# PLAY FINAL VIDEO
# =========================================================

@app.route("/projects/<int:project_id>/final-video")
def play_project_final_video(project_id):

    project = get_project_or_404(project_id)

    if not project or not project.get("final_video_path"):
        return "Final video not found", 404

    try:

        video_path = get_safe_project_file(
            project["final_video_path"]
        )

        validate_generated_file(
            video_path,
            "Final video",
        )

        return send_file(
            video_path,
            mimetype="video/mp4",
            conditional=True,
        )

    except ValueError:
        return "Invalid video file path", 400

    except FileNotFoundError:
        return "Final video file not found", 404


# =========================================================
# DOWNLOAD FINAL VIDEO
# =========================================================

@app.route(
    "/projects/<int:project_id>/download-final-video"
)
def download_project_final_video(project_id):

    project = get_project_or_404(project_id)

    if not project or not project.get("final_video_path"):
        return "Final video not found", 404

    try:

        video_path = get_safe_project_file(
            project["final_video_path"]
        )

        validate_generated_file(
            video_path,
            "Final video",
        )

        return send_file(
            video_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name=f"cineforge_project_{project_id}.mp4",
        )

    except ValueError:
        return "Invalid video file path", 400

    except FileNotFoundError:
        return "Final video file not found", 404


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=False,
    )