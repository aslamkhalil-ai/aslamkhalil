
from pathlib import Path

from database import get_project
from engines.composition_engine import create_final_video
from engines.music_manager import get_music_for_style

PROJECT_ID = 9

root = Path(__file__).resolve().parent
project = get_project(PROJECT_ID)

if not project:
    raise RuntimeError("Project not found.")

project_dir = root / "projects" / str(PROJECT_ID)

video = project_dir / "scene_video.mp4"
narration = project_dir / "narration.wav"
subtitles = project_dir / "subtitles.srt"
output = project_dir / "final_video.mp4"

for file in [video, narration, subtitles]:
    if not file.is_file():
        raise FileNotFoundError(f"Missing file: {file}")

music = get_music_for_style(
    project.get("genre", "ambient")
)

print("Rendering final video with corrected subtitles...")

result = create_final_video(
    str(video),
    str(narration),
    str(subtitles),
    str(output),
    music_path=music
)

print("\nSUCCESS!")
print(f"Final video: {result}")
print("AI visuals, narration and subtitles source preserved.")
