
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline

# CPU optimization
torch.set_num_threads(2)

MODEL_PATH = "models/stable-diffusion-v1-5"
OUTPUT_PATH = Path("models/test_cinematic_image.png")

print("Loading Stable Diffusion 1.5...")
print("Device: CPU")
print("Please wait. First loading may take several minutes.")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    use_safetensors=True,
    local_files_only=True,
    safety_checker=None,
    feature_extractor=None
)

pipe = pipe.to("cpu")

# Reduce memory usage

pipe.enable_attention_slicing()
prompt = """
A cinematic photorealistic science fiction scene,
a lone astronaut standing on an alien planet,
enormous mysterious spacecraft in the distant sky,
dramatic cinematic lighting, realistic textures,
volumetric fog, atmospheric perspective,
professional Hollywood movie cinematography,
high detail, wide composition, 16:9
"""

print("Generating first AI image...")


image = pipe(
    prompt=prompt,
    negative_prompt=(
        "blurry, distorted, deformed astronaut, malformed body, "
        "low quality, cartoon, illustration, text, watermark, "
        "duplicate objects, unrealistic proportions"
    ),
    height=512,
    width=512,
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]


image.save(OUTPUT_PATH)

print(f"SUCCESS! Image saved at: {OUTPUT_PATH}")
