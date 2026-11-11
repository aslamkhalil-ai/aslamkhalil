"""CineForge AI - Regenerate Visuals Without Recreating Audio."""

import argparse
import json
import logging
import sys
from pathlib import Path

import app as cine_app

from database import get_project, update_project

from engines.storyboard_engine import (
    create_storyboard_image,
    select_camera_motion,
)

from engines.visual_engine import (
    create_scene_clip,
    combine_scene_clips,
)

from engines.composition_engine import create_final_video
from engines.music_manager import get_music_for_style


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("CineForge.VisualRegenerator")

ROOT = Path(cine_app.app.root_path).resolve()


def validate_file(path, label):
    """Ensure a generated file exists and is not empty."""

    path = Path(path)

    if not path.is_file():
        raise RuntimeError(f"{label} was not created: {path}")

    if path.stat().st_size == 0:
        raise RuntimeError(f"{label} is empty: {path}")

    return path


def resolve_project_file(root, relative_path, fallback):
    """Resolve a project file while keeping it inside the project root."""

    selected = Path(relative_path) if relative_path else Path(fallback)

    if selected.is_absolute():
        raise ValueError("Absolute project file paths are not allowed.")

    resolved = (root / selected).resolve()

    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Project file path is outside the application folder.")

    return resolved


def main(project_id):
    project = get_project(project_id)

    if not project:
        raise RuntimeError(f"Project {project_id} not found.")

    project_dir = ROOT / "projects" / str(project_id)
    project_dir.mkdir(parents=True, exist_ok=True)

    voice_file = resolve_project_file(
        ROOT,
        project.get("voice_path"),
        f"projects/{project_id}/narration.wav"
    )

    subtitle_file = resolve_project_file(
        ROOT,
        project.get("subtitle_path"),
        f"projects/{project_id}/subtitles.srt"
    )

    validate_file(voice_file, "Narration")
    validate_file(subtitle_file, "Subtitle file")

    scenes = json.loads(project.get("scenes") or "[]")

    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError("Project has no valid scenes.")

    narration_duration = cine_app._audio_duration_seconds(voice_file)

    if narration_duration <= 0:
        raise RuntimeError("Narration duration is invalid.")

    durations = cine_app._scene_durations(
        scenes,
        narration_duration
    )

    if len(durations) != len(scenes):
        raise RuntimeError("Scene duration calculation failed.")

    visuals_dir = project_dir / "scene_visuals"
    clips_dir = project_dir / "scene_clips"

    visuals_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = []

    logger.info("Starting visual regeneration")
    logger.info("Project: %s", project.get("title"))
    logger.info("Scene count: %s", len(scenes))
    logger.info("Narration duration: %.2f seconds", narration_duration)
    logger.info("Existing narration and subtitles will be preserved.")

    try:
        # Generate new artwork and cinematic scene clips.
        for index, scene in enumerate(scenes):

            if not isinstance(scene, dict):
                raise ValueError(
                    f"Invalid scene data at scene {index + 1}."
                )

            image_path = visuals_dir / f"scene_{index + 1:02d}.jpg"
            clip_path = clips_dir / f"scene_{index + 1:02d}.mp4"

            logger.info(
                "Generating scene %s/%s",
                index + 1,
                len(scenes)
            )

            create_storyboard_image(
                scene,
                image_path,
                project.get("title", "CineForge Production"),
                index,
                len(scenes)
            )

            validate_file(image_path, f"Scene {index + 1} image")

            motion = select_camera_motion(scene, index)

            logger.info(
                "Scene %s | Motion: %s | Duration: %.2fs",
                index + 1,
                motion,
                durations[index]
            )

            create_scene_clip(
                str(image_path),
                str(clip_path),
                duration=durations[index],
                motion=motion
            )

            validate_file(clip_path, f"Scene {index + 1} clip")

            clip_paths.append(str(clip_path))

        # Combine the newly generated scene clips.
        visual_rel = f"projects/{project_id}/scene_video.mp4"
        visual_file = project_dir / "scene_video.mp4"

        logger.info("Combining scene clips...")

        combine_scene_clips(
            clip_paths,
            str(visual_file)
        )

        validate_file(visual_file, "Combined scene video")

        # Background music is optional.
        music_file = None

        try:
            music_file = get_music_for_style(
                project.get("genre", "ambient")
            )
        except (FileNotFoundError, OSError) as exc:
            logger.warning("Background music unavailable: %s", exc)

        # Render the final video using existing narration and subtitles.
        final_rel = f"projects/{project_id}/final_video.mp4"
        final_file = project_dir / "final_video.mp4"

        logger.info("Rendering final video...")

        create_final_video(
            str(visual_file),
            str(voice_file),
            str(subtitle_file),
            str(final_file),
            music_path=music_file
        )

        validate_file(final_file, "Final video")

        update_project(
            project_id,
            visual_video_path=visual_rel,
            final_video_path=final_rel,
            status="Video ready",
            error=""
        )

        logger.info("CINEFORGE AI PRODUCTION COMPLETED")
        logger.info("Final video: %s", final_file)
        logger.info("Original narration and subtitles preserved.")

        return str(final_file)

    except Exception as exc:
        logger.exception("Visual regeneration failed.")

        try:
            update_project(
                project_id,
                status="Generation failed",
                error=str(exc)[:2000]
            )
        except Exception:
            logger.exception("Could not update project error status.")

        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate CineForge project visuals."
    )

    parser.add_argument(
        "--project",
        type=int,
        default=9,
        help="Project ID to regenerate (default: 9)"
    )

    args = parser.parse_args()

    try:
        main(args.project)
    except Exception as exc:
        logger.error("Production stopped: %s", exc)
        sys.exit(1)