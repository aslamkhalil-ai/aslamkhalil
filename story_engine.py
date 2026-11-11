
"""
CineForge AI - Professional Story & Script Generation Engine

Local-first cinematic story generation using Ollama.
Features:
- Duration-aware narration planning
- Scene-level narration validation
- Consistent scene structure
- Automatic correction retry
- Strict JSON validation
- No paid APIs or cloud inference
"""

import os
import json
import logging
import re
import requests

from typing import Any


logger = logging.getLogger(__name__)


# ==================================================
# CONFIGURATION
# ==================================================

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
).rstrip("/")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:latest"
)

REQUEST_TIMEOUT = (10, 600)

DEFAULT_OPTIONS = {
    "temperature": 0.55,
    "top_p": 0.9,
    "repeat_penalty": 1.12,
    "num_ctx": 4096,
    "num_predict": 2800
}

# Natural narration speed.
# 2.2 words per second is a practical cinematic target.
WORDS_PER_SECOND = 2.2

# Allow a small tolerance for natural speech pacing.
MIN_WORD_RATIO = 0.88
MAX_WORD_RATIO = 1.20

MAX_GENERATION_ATTEMPTS = 2


# ==================================================
# TEXT UTILITIES
# ==================================================

def count_words(text: str) -> int:
    """Count spoken words in narration."""

    if not isinstance(text, str):
        return 0

    return len(
        re.findall(
            r"\b[\w']+(?:[-'][\w]+)*\b",
            text,
            flags=re.UNICODE
        )
    )


def expected_word_range(duration: int) -> tuple[int, int]:
    """Return acceptable narration word range."""

    target = duration * WORDS_PER_SECOND

    minimum = max(
        1,
        int(target * MIN_WORD_RATIO)
    )

    maximum = int(
        target * MAX_WORD_RATIO
    )

    return minimum, maximum


def clean_json_response(raw: str) -> dict:
    """Safely parse an Ollama JSON response."""

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Empty AI response.")

    raw = raw.strip()

    # Remove accidental markdown fences.
    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw,
        flags=re.IGNORECASE
    )

    raw = re.sub(
        r"\s*```$",
        "",
        raw
    )

    try:
        return json.loads(raw)

    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")

        if start == -1 or end <= start:
            raise ValueError(
                "AI did not return a valid JSON object."
            )

        return json.loads(
            raw[start:end + 1]
        )


# ==================================================
# SCENE PLANNING
# ==================================================

def calculate_scene_count(duration: int) -> int:
    """Determine scene count based on target runtime."""

    if duration <= 30:
        return 3

    if duration <= 45:
        return 4

    if duration <= 60:
        return 5

    if duration <= 90:
        return 6

    return min(
        8,
        max(6, round(duration / 15))
    )


def calculate_scene_durations(
    duration: int,
    scene_count: int
) -> list[int]:
    """Distribute total runtime across scenes."""

    base = duration // scene_count
    remainder = duration % scene_count

    return [
        base + (1 if index < remainder else 0)
        for index in range(scene_count)
    ]


# ==================================================
# PROMPT GENERATION
# ==================================================

def build_story_prompt(
    idea: str,
    genre: str,
    duration: int,
    scene_count: int,
    durations: list[int],
    correction_feedback: str = ""
) -> str:

    scene_duration_text = "\n".join(
        f"Scene {index + 1}: {seconds} seconds, "
        f"approximately {round(seconds * WORDS_PER_SECOND)} spoken words"
        for index, seconds in enumerate(durations)
    )

    minimum_words, maximum_words = expected_word_range(
        duration
    )

    return f"""
You are a professional film director, screenwriter,
cinematographer and cinematic trailer producer.

Create a production-ready cinematic video plan
for CINEFORGE AI.

PROJECT GENRE:
{genre}

CREATIVE BRIEF:
{idea}

TOTAL RUNTIME:
Exactly {duration} seconds.

SCENE COUNT:
Exactly {scene_count} scenes.

SCENE TIMING AND NARRATION TARGETS:
{scene_duration_text}

TOTAL NARRATION WORD REQUIREMENT:
Minimum {minimum_words} words.
Maximum {maximum_words} words.
Ideal target approximately {duration * WORDS_PER_SECOND:.0f} words.

CRITICAL NARRATION REQUIREMENTS:

1. Narration must fill the complete video naturally.
2. Do not write an extremely short summary.
3. Write cinematic, expressive, natural spoken narration.
4. Every scene must contain enough narration for its duration.
5. Use punctuation and natural pauses.
6. Do not repeat the same sentence in different scenes.
7. Do not add stage directions inside spoken narration.
8. The scene narration fields together form the complete voice script.
9. Keep narration coherent from beginning to end.
10. Avoid unnatural filler just to increase word count.

STORY STRUCTURE:

Opening:
- Immediately establish an intriguing visual hook.
- Introduce the central subject or mystery.

Development:
- Build tension and curiosity.
- Introduce meaningful events.
- Maintain continuity between scenes.

Climax:
- Deliver the strongest dramatic moment.

Ending:
- Provide a memorable conclusion or cinematic reveal.
- End with a meaningful final visual.

VISUAL CONTINUITY:

- Maintain the same main character identity.
- Keep clothing, age, appearance and environment consistent.
- Maintain logical location continuity.
- Give each scene a distinct action.
- Avoid unrelated imagery.
- Describe a single uninterrupted cinematic frame.
- Use realistic foreground, middle ground and background.
- Specify subject placement and camera framing.
- Include cinematic lighting, atmosphere and depth.

AI IMAGE PROMPT REQUIREMENTS:

Every visual_prompt must independently describe one
complete cinematic 16:9 image.

Include:
- Main subject and physical appearance.
- Specific action.
- Location and environment.
- Camera framing and perspective.
- Foreground, middle ground and background.
- Lighting and atmospheric conditions.
- Realistic material textures.
- Genre-specific details.
- Cinematic color grading.
- Professional film still aesthetic.

STRICT VISUAL RESTRICTIONS:

- One scene only.
- One continuous full-frame composition.
- No collage.
- No split screen.
- No contact sheets.
- No multiple panels.
- No comic book layout.
- No storyboard grids.
- No captions.
- No logos.
- No watermarks.
- No written text.
- No presentation slides.

CAMERA DIRECTION:

Specify:
- Shot type.
- Camera angle.
- Lens or perspective where useful.
- Camera movement.
- Subject movement.

AUDIO DIRECTION:

Describe:
- Background ambience.
- Environmental sound.
- Important sound effects.
- Emotional atmosphere.
- Suitable intensity and timing.

Do not invent audio files or claim that sound effects
have been rendered.

RETURN ONLY VALID JSON.

EXACT JSON STRUCTURE:

{{
    "title": "Professional cinematic title",
    "logline": "One compelling sentence",
    "script": "Complete narration assembled from all scenes",
    "scenes": [
        {{
            "scene_number": 1,
            "duration_seconds": 12,
            "visual_description": "Detailed visual description",
            "camera_direction": "Shot type, angle and movement",
            "narration": "Complete spoken narration for this scene",
            "sound_design": "Atmosphere and sound effect direction",
            "visual_prompt": "Detailed standalone 16:9 cinematic image prompt"
        }}
    ]
}}

STRICT VALIDATION RULES:

- Exactly {scene_count} scenes.
- Exact scene durations as specified.
- Scene numbering starts at 1.
- All fields must contain meaningful text.
- The total scene duration must equal {duration}.
- Narration must contain {minimum_words} to {maximum_words} words.
- Script must match the combined scene narration.
- Do not return markdown.
- Do not return explanations.
- Do not omit any JSON fields.

CORRECTION FEEDBACK:
{correction_feedback or "No previous correction. Generate the complete story."}
"""


# ==================================================
# RESPONSE VALIDATION
# ==================================================

def validate_story(
    data: dict[str, Any],
    duration: int,
    scene_count: int
) -> dict[str, Any]:

    if not isinstance(data, dict):
        raise ValueError("AI response must be a JSON object.")

    title = data.get("title", "")
    logline = data.get("logline", "")
    scenes = data.get("scenes", [])

    if not isinstance(title, str) or not title.strip():
        raise ValueError("Generated title is missing.")

    if not isinstance(logline, str) or not logline.strip():
        raise ValueError("Generated logline is missing.")

    if not isinstance(scenes, list):
        raise ValueError("Scenes must be a list.")

    if len(scenes) != scene_count:
        raise ValueError(
            f"Expected {scene_count} scenes, got {len(scenes)}."
        )

    expected_durations = calculate_scene_durations(
        duration,
        scene_count
    )

    required_fields = [
        "visual_description",
        "camera_direction",
        "narration",
        "sound_design",
        "visual_prompt"
    ]

    validated_scenes = []

    for index, scene in enumerate(scenes):

        number = index + 1

        if not isinstance(scene, dict):
            raise ValueError(
                f"Scene {number} is not a valid object."
            )

        for field in required_fields:

            value = scene.get(field)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Scene {number}: missing {field}."
                )

        try:
            scene_duration = int(
                scene.get("duration_seconds")
            )

        except (TypeError, ValueError):
            raise ValueError(
                f"Scene {number}: invalid duration."
            )

        if scene_duration != expected_durations[index]:
            raise ValueError(
                f"Scene {number} must be "
                f"{expected_durations[index]} seconds."
            )

        narration = scene["narration"].strip()

        scene_minimum, scene_maximum = expected_word_range(
            scene_duration
        )

        scene_words = count_words(narration)

        if not scene_minimum <= scene_words <= scene_maximum:
            raise ValueError(
                f"Scene {number} narration has {scene_words} words. "
                f"Expected {scene_minimum}-{scene_maximum} words "
                f"for {scene_duration} seconds."
            )

        validated_scenes.append({
            "scene_number": number,
            "duration_seconds": scene_duration,
            "visual_description": scene["visual_description"].strip(),
            "camera_direction": scene["camera_direction"].strip(),
            "narration": narration,
            "sound_design": scene["sound_design"].strip(),
            "visual_prompt": scene["visual_prompt"].strip()
        })

    if sum(
        item["duration_seconds"]
        for item in validated_scenes
    ) != duration:
        raise ValueError("Total scene duration is incorrect.")

    # The actual spoken script is assembled from scenes.
    combined_script = " ".join(
        scene["narration"]
        for scene in validated_scenes
    )

    minimum_words, maximum_words = expected_word_range(
        duration
    )

    total_words = count_words(combined_script)

    if not minimum_words <= total_words <= maximum_words:
        raise ValueError(
            f"Total narration has {total_words} words. "
            f"Expected {minimum_words}-{maximum_words}."
        )

    return {
        "title": title.strip(),
        "logline": logline.strip(),
        "script": combined_script,
        "scenes": validated_scenes
    }


# ==================================================
# OLLAMA GENERATION
# ==================================================

def _request_ollama(prompt: str) -> dict:

    endpoint = f"{OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "5m",
        "options": DEFAULT_OPTIONS
    }

    try:

        response = requests.post(
            endpoint,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        result = response.json()

        raw_response = result.get(
            "response",
            ""
        )

        return clean_json_response(raw_response)

    except requests.exceptions.Timeout as error:

        raise RuntimeError(
            "Story generation timed out. "
            "The local model may need more CPU time."
        ) from error

    except requests.exceptions.ConnectionError as error:

        raise RuntimeError(
            "Cannot connect to Ollama. "
            "Make sure Ollama is running."
        ) from error

    except requests.exceptions.HTTPError as error:

        detail = ""

        if error.response is not None:
            detail = error.response.text[:300]

        raise RuntimeError(
            f"Ollama request failed: {detail or str(error)}"
        ) from error

    except requests.exceptions.RequestException as error:

        raise RuntimeError(
            f"Local AI request failed: {error}"
        ) from error


# ==================================================
# MAIN STORY GENERATION
# ==================================================

def generate_story(
    idea: str,
    genre: str,
    duration: int
) -> dict:

    idea = str(idea).strip()
    genre = str(genre).strip()

    if not idea:
        raise ValueError("Please provide a creative idea.")

    if not genre:
        raise ValueError("Please select a video genre.")

    try:
        duration = int(duration)

    except (TypeError, ValueError) as error:
        raise ValueError(
            "Video duration must be a valid number."
        ) from error

    if duration < 15 or duration > 180:
        raise ValueError(
            "Video duration must be between 15 and 180 seconds."
        )

    scene_count = calculate_scene_count(duration)

    scene_durations = calculate_scene_durations(
        duration,
        scene_count
    )

    correction_feedback = ""

    logger.info(
        "Starting local story generation: %s seconds, %s scenes.",
        duration,
        scene_count
    )

    last_error = None

    for attempt in range(MAX_GENERATION_ATTEMPTS):

        prompt = build_story_prompt(
            idea=idea,
            genre=genre,
            duration=duration,
            scene_count=scene_count,
            durations=scene_durations,
            correction_feedback=correction_feedback
        )

        try:

            raw_data = _request_ollama(prompt)

            story = validate_story(
                data=raw_data,
                duration=duration,
                scene_count=scene_count
            )

            logger.info(
                "Story successfully generated: %s | %s words",
                story["title"],
                count_words(story["script"])
            )

            return {
                "title": story["title"],
                "logline": story["logline"],
                "script": story["script"],
                "scenes": json.dumps(
                    story["scenes"],
                    indent=2,
                    ensure_ascii=False
                )
            }

        except (ValueError, RuntimeError) as error:

            last_error = error

            logger.warning(
                "Story generation attempt %s failed: %s",
                attempt + 1,
                error
            )

            correction_feedback = (
                f"Your previous response failed validation: {error}. "
                "Regenerate the entire JSON. "
                "Correct every issue. "
                "Do not shorten narration. "
                "Follow all scene-level word targets exactly."
            )

    raise RuntimeError(
        "Could not generate a valid production-ready story "
        f"after {MAX_GENERATION_ATTEMPTS} attempts. "
        f"Last error: {last_error}"
    )
