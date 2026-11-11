
"""
CineForge AI - Local AI Cinematic Storyboard Engine

Offline cinematic image generation using Stable Diffusion 1.5.
Optimized for CPU-based systems and 16:9 video production.
"""

import os
import gc
import logging
import threading

from pathlib import Path

import torch

from PIL import (
    Image,
    ImageEnhance,
    ImageFilter,
    ImageOps,
    ImageDraw,
)

from diffusers import StableDiffusionPipeline


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

logger = logging.getLogger(__name__)

WIDTH = 1280
HEIGHT = 720

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = Path(
    os.getenv(
        "CINEFORGE_SD_MODEL",
        str(PROJECT_ROOT / "models" / "stable-diffusion-v1-5")
    )
)

# Native widescreen generation.
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360

INFERENCE_STEPS = 18
GUIDANCE_SCALE = 7.0
CPU_THREADS = 2

_pipeline = None
_model_lock = threading.Lock()


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

def _load_pipeline():

    global _pipeline

    with _model_lock:

        if _pipeline is not None:
            return _pipeline

        if not MODEL_PATH.is_dir():
            raise FileNotFoundError(
                f"Stable Diffusion model not found: {MODEL_PATH.resolve()}"
            )

        if not (MODEL_PATH / "model_index.json").is_file():
            raise FileNotFoundError(
                f"Invalid Stable Diffusion model directory: {MODEL_PATH}"
            )

        try:

            torch.set_num_threads(CPU_THREADS)

            logger.info("Loading local Stable Diffusion 1.5...")

            pipe = StableDiffusionPipeline.from_pretrained(
                str(MODEL_PATH),
                torch_dtype=torch.float32,
                use_safetensors=True,
                local_files_only=True,
                safety_checker=None,
                feature_extractor=None,
            )

            pipe = pipe.to("cpu")

            pipe.enable_attention_slicing("max")

            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()

            pipe.set_progress_bar_config(disable=True)

            _pipeline = pipe

            logger.info("Stable Diffusion loaded successfully.")

            return _pipeline

        except Exception:

            _pipeline = None
            gc.collect()

            logger.exception("Stable Diffusion initialization failed.")

            raise


# ---------------------------------------------------------
# SCENE CLASSIFICATION
# ---------------------------------------------------------

SCENE_KEYWORDS = {

    "space": (
        "astronaut", "spaceship", "spacecraft", "space station",
        "galaxy", "planet", "orbit", "cosmic", "spacesuit",
        "alien", "moon", "nebula", "universe", "star system"
    ),

    "horror": (
        "horror", "ghost", "haunted", "darkness", "monster",
        "abandoned", "creature", "shadow", "graveyard",
        "terrifying", "nightmare", "demon", "eerie",
        "possessed", "supernatural"
    ),

    "city": (
        "city", "street", "building", "urban", "skyscraper",
        "traffic", "neon", "downtown", "metropolis",
        "apartment", "highway", "rooftop"
    ),

    "forest": (
        "forest", "jungle", "woods", "trees", "mountain",
        "river", "wilderness", "nature", "waterfall",
        "valley", "lake", "woodland"
    ),

    "historical": (
        "castle", "kingdom", "medieval", "ancient", "warrior",
        "historical", "empire", "temple", "palace", "battle",
        "king", "queen", "sword", "dynasty", "village"
    ),

    "technology": (
        "hologram", "computer", "control room", "robot",
        "technology", "laboratory", "machine", "digital",
        "interface", "android", "cyber", "artificial intelligence",
        "server room"
    ),
}


def _classify_scene(description, preferred_genre=None):

    text = str(description or "").lower()
    genre = str(preferred_genre or "").lower().strip()

    aliases = {
        "science fiction": "space",
        "sci-fi": "space",
        "sci fi": "space",
        "suspense": "cinematic",
        "mystery": "cinematic",
        "historical fiction": "historical",
        "history": "historical",
        "urban": "city",
        "nature": "forest",
        "horror": "horror",
    }

    genre = aliases.get(genre, genre)

    if genre in SCENE_KEYWORDS:
        return genre

    for category, keywords in SCENE_KEYWORDS.items():

        if any(keyword in text for keyword in keywords):
            return category

    return "cinematic"


# ---------------------------------------------------------
# CAMERA MOVEMENT
# ---------------------------------------------------------

def select_camera_motion(scene, scene_index=0):

    if isinstance(scene, dict):

        description = " ".join(
            str(scene.get(key, ""))
            for key in (
                "visual_description",
                "visual_prompt",
                "description",
                "camera_direction",
            )
        )

        camera_direction = str(
            scene.get("camera_direction", "")
        ).lower()

        genre = scene.get("genre", "")

    else:

        description = str(scene or "")
        camera_direction = ""
        genre = ""

    directions = (
        ("zoom in", "zoom_in"),
        ("push in", "zoom_in"),
        ("zoom out", "zoom_out"),
        ("pull back", "zoom_out"),
        ("pan left", "pan_left"),
        ("pan right", "pan_right"),
        ("pan up", "pan_up"),
        ("pan down", "pan_down"),
    )

    for keyword, movement in directions:

        if keyword in camera_direction:
            return movement

    if "static" in camera_direction or "locked camera" in camera_direction:
        return "static"

    category = _classify_scene(description, genre)

    movements = {

        "space": [
            "zoom_in", "pan_left", "zoom_out", "pan_right"
        ],

        "horror": [
            "zoom_in", "static", "pan_left", "zoom_in"
        ],

        "city": [
            "pan_left", "pan_right", "zoom_in", "zoom_out"
        ],

        "forest": [
            "zoom_in", "pan_up", "pan_right", "zoom_out"
        ],

        "historical": [
            "zoom_out", "pan_left", "zoom_in", "pan_right"
        ],

        "technology": [
            "zoom_in", "pan_right", "static", "zoom_out"
        ],

        "cinematic": [
            "zoom_in", "zoom_out", "pan_left", "pan_right"
        ],
    }

    options = movements.get(
        category,
        movements["cinematic"]
    )

    return options[scene_index % len(options)]


# ---------------------------------------------------------
# GENRE ART DIRECTION
# ---------------------------------------------------------

GENRE_STYLES = {

    "space": (
        "photorealistic science fiction feature film, "
        "realistic spacecraft engineering, detailed astronaut "
        "spacesuits, enormous celestial bodies, cosmic scale, "
        "deep atmospheric space, dramatic cinematic lighting"
    ),

    "horror": (
        "photorealistic psychological horror feature film, "
        "deep natural shadows, unsettling atmosphere, "
        "eerie environments, practical cinematic lighting, "
        "dramatic darkness, realistic suspense cinematography"
    ),

    "city": (
        "photorealistic urban feature film cinematography, "
        "detailed architecture, realistic streets, "
        "atmospheric lighting, natural reflections, "
        "detailed cinematic city environments"
    ),

    "forest": (
        "photorealistic nature cinematography, "
        "dense realistic wilderness, detailed vegetation, "
        "atmospheric fog, natural sunlight, "
        "cinematic environmental depth"
    ),

    "historical": (
        "photorealistic historical epic cinematography, "
        "period-inspired architecture, authentic-looking "
        "costumes, detailed historical environments, "
        "natural lighting, epic cinematic composition"
    ),

    "technology": (
        "photorealistic futuristic science fiction film, "
        "advanced machinery, detailed control systems, "
        "realistic technological materials, atmospheric "
        "lighting, cinematic production design"
    ),

    "cinematic": (
        "photorealistic professional feature film, "
        "realistic environments, detailed production design, "
        "dramatic natural lighting, cinematic composition"
    ),
}


# ---------------------------------------------------------
# PROMPT GENERATION
# ---------------------------------------------------------

def _build_prompt(scene, project_title, category):

    if isinstance(scene, dict):

        description = (
            scene.get("visual_prompt")
            or scene.get("visual_description")
            or scene.get("description")
            or "A mysterious cinematic scene"
        )

        action = scene.get("action", "")
        setting = scene.get("setting", "")
        mood = scene.get("mood", "")
        camera = scene.get("camera_direction", "")

    else:

        description = str(scene or "A cinematic scene")

        action = ""
        setting = ""
        mood = ""
        camera = ""

    description = str(description).strip()[:900]

    additional_details = " ".join(
        str(item).strip()
        for item in (action, setting, mood, camera)
        if item
    )

    visual_style = GENRE_STYLES.get(
        category,
        GENRE_STYLES["cinematic"]
    )

    prompt = f"""
A single uninterrupted widescreen 16:9 cinematic movie frame.

FILM:
{project_title}

SCENE DESCRIPTION:
{description}

SCENE DETAILS:
{additional_details}

ART DIRECTION:
{visual_style}

COMPOSITION:

One coherent physical location.
One main visual subject.
One continuous camera viewpoint.
Wide cinematic composition.
Clear foreground, middle ground and background.
Natural perspective and realistic depth.
Professional feature film cinematography.
Carefully controlled cinematic lighting.
Realistic environmental textures.
Natural atmospheric perspective.
Detailed but believable production design.

IMAGE QUALITY:

Photorealistic film still.
Highly detailed realistic environment.
Sharp primary subject.
Natural material surfaces.
Realistic shadows and reflections.
Balanced cinematic color grading.
Professional visual storytelling.

STRICT COMPOSITION RULES:

Create exactly ONE image of ONE scene.
Fill the entire widescreen frame.
Do not create a storyboard.
Do not create multiple panels.
Do not create a split screen.
Do not create a collage or contact sheet.
Do not place multiple separate scenes in the image.
Do not add borders or frames.

No text, titles, subtitles, captions, logos or watermarks.
"""

    return " ".join(prompt.split())


# ---------------------------------------------------------
# NEGATIVE PROMPT
# ---------------------------------------------------------

NEGATIVE_PROMPT = """
blurry, out of focus, pixelated, low resolution,
low quality, jpeg artifacts, distorted image,
cartoon, anime, illustration, painting, sketch,
storyboard, comic panels, multiple panels, split screen,
collage, contact sheet, grid layout, multiple scenes,
multiple frames, border, frame, picture within picture,
text, letters, typography, subtitles, watermark, logo,
user interface, deformed human, malformed hands,
extra fingers, missing fingers, duplicate limbs,
duplicate people, duplicate objects, distorted face,
unrealistic anatomy, bad proportions, plastic skin,
artificial textures, oversaturated colors, flat lighting,
washed out colors, poor composition, poorly rendered
background, unnatural perspective, inconsistent lighting,
floating objects, malformed architecture
"""


# ---------------------------------------------------------
# SUBTITLE GRADIENT
# ---------------------------------------------------------

def _create_subtitle_gradient(width, height):

    overlay = Image.new(
        "RGBA",
        (width, height),
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(overlay)

    gradient_start = int(height * 0.72)

    for y in range(gradient_start, height):

        progress = (
            (y - gradient_start)
            / max(1, height - gradient_start)
        )

        alpha = int(145 * (progress ** 1.7))

        draw.line(
            (0, y, width, y),
            fill=(0, 0, 0, alpha)
        )

    return overlay


# ---------------------------------------------------------
# CINEMATIC FRAME PROCESSING
# ---------------------------------------------------------

def _prepare_cinematic_frame(image, output_path):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    image = image.convert("RGB")

    # Preserve the generated 16:9 composition.
    image = ImageOps.fit(
        image,
        (WIDTH, HEIGHT),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5)
    )

    image = ImageEnhance.Contrast(image).enhance(1.06)
    image = ImageEnhance.Color(image).enhance(1.04)

    # Moderate sharpening avoids excessive artificial edges.
    image = ImageEnhance.Sharpness(image).enhance(1.10)

    overlay = _create_subtitle_gradient(
        WIDTH,
        HEIGHT
    )

    image = Image.alpha_composite(
        image.convert("RGBA"),
        overlay
    ).convert("RGB")

    image = image.filter(
        ImageFilter.UnsharpMask(
            radius=1.0,
            percent=45,
            threshold=4
        )
    )

    image.save(
        output_path,
        format="JPEG",
        quality=96,
        optimize=True,
        subsampling=0
    )

    return str(output_path.resolve())


# ---------------------------------------------------------
# MAIN STORYBOARD GENERATION
# ---------------------------------------------------------

def create_storyboard_image(
    scene,
    output_path,
    project_title,
    scene_index,
    total_scenes
):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if isinstance(scene, dict):

        scene_number = int(
            scene.get("scene_number", scene_index + 1)
        )

        description = (
            scene.get("visual_description")
            or scene.get("visual_prompt")
            or scene.get("description")
            or "A cinematic scene"
        )

        genre = str(scene.get("genre", "")).lower()

    else:

        scene_number = scene_index + 1
        description = str(scene or "A cinematic scene")
        genre = ""

    category = _classify_scene(
        f"{genre} {description}",
        genre
    )

    prompt = _build_prompt(
        scene,
        project_title,
        category
    )

    seed = (
        scene_number * 7919
        + sum(ord(char) for char in str(project_title))
    )

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(seed)

    logger.info(
        "Generating AI scene %s/%s (%s)",
        scene_index + 1,
        total_scenes,
        category
    )

    pipe = _load_pipeline()

    try:

        with _model_lock:

            with torch.inference_mode():

                result = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    height=IMAGE_HEIGHT,
                    width=IMAGE_WIDTH,
                    num_inference_steps=INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=generator,
                    num_images_per_prompt=1,
                )

            generated_image = result.images[0]

            saved_path = _prepare_cinematic_frame(
                generated_image,
                output_path
            )

            del generated_image
            del result

            gc.collect()

        logger.info(
            "Cinematic scene saved: %s",
            saved_path
        )

        return saved_path

    except Exception:

        logger.exception(
            "AI image generation failed for scene %s.",
            scene_number
        )

        gc.collect()

        raise
