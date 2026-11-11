"""CineForge AI - Genre-Aware Background Music Manager."""

import random
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MUSIC_DIR = BASE_DIR / "assets" / "music"

SUPPORTED_AUDIO_FORMATS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".ogg",
    ".flac"
}

MUSIC_CATEGORIES = {
    "cinematic": "cinematic",
    "promotional": "cinematic",
    "trailer": "cinematic",
    "game trailer": "cinematic",
    "science fiction": "cinematic",
    "sci-fi": "cinematic",
    "scifi": "cinematic",
    "space": "cinematic",
    "historical": "cinematic",

    "horror": "horror",
    "suspense": "horror",
    "thriller": "horror",
    "dark": "horror",

    "documentary": "documentary",
    "educational": "ambient",

    "ambient": "ambient",
    "mystery": "ambient",
    "drama": "ambient",
    "emotional": "ambient",
    "nature": "ambient"
}



def normalize_style(style):
    """Normalize a genre or style name."""

    if not style:
        return "ambient"

    normalized = str(style).strip().lower()
    normalized = normalized.replace("_", " ")
    normalized = normalized.replace("-", " ")

    return " ".join(normalized.split())


def get_music_category(style):
    """Return the matching music category."""

    normalized = normalize_style(style)

    return MUSIC_CATEGORIES.get(normalized, "ambient")


def get_music_files(category):
    """Find supported audio files inside a category folder."""

    category_folder = MUSIC_DIR / category

    if not category_folder.is_dir():
        return []

    return sorted(
        file_path
        for file_path in category_folder.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_AUDIO_FORMATS
    )


def get_music_for_style(style, fallback=True):
    """
    Select background music based on the project genre.

    Returns:
        str: Selected audio file path.

    Raises:
        FileNotFoundError: If no suitable music is available
        and fallback is disabled.
    """

    category = get_music_category(style)
    music_files = get_music_files(category)

    # If the selected category is empty, optionally use ambient music.
    if not music_files and fallback and category != "ambient":
        logger.warning(
            "No music found in '%s'. Trying ambient fallback.",
            category
        )

        category = "ambient"
        music_files = get_music_files(category)

    if not music_files:
        message = (
            f"No supported background music found for style '{style}'. "
            f"Checked directory: {MUSIC_DIR / category}"
        )

        if fallback:
            logger.warning(message)
            return None

        raise FileNotFoundError(message)

    selected_music = random.choice(music_files)

    logger.info("Music category: %s", category)
    logger.info("Selected music: %s", selected_music.name)

    return str(selected_music)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    music = get_music_for_style("cinematic")

    if music:
        print("\nMusic selection successful!")
        print(music)
    else:
        print("\nNo background music available.")
        print("Add audio files inside assets/music category folders.")