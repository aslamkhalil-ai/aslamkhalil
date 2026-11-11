from pathlib import Path

from engines.composition_engine import create_final_video
from engines.music_manager import get_music_for_style
from database import get_project

project_id = 9
project = get_project(project_id)

root = Path(__file__).resolve().parent

video = root / f"projects/{project_id}/scene_video.mp4"
voice = root / f"projects/{project_id}/narration.wav"
subtitles = root / f"projects/{project_id}/subtitles.srt"
output = root / f"projects/{project_id}/final_video.mp4"

music = get_music_for_style(project.get("genre", "ambient"))

create_final_video(
    str(video),
    str(voice),
    str(subtitles),
    str(output),
    music_path=music
)

print("Final video rendered:", output)