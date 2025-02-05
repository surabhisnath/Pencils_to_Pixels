import os
from openai import OpenAI
from PIL import Image
import requests

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

for stim in ["G", "I", "R"]:
    image = Image.open(f"../../stimuli/originals/stimuli_{stim}.png")
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")
        image.save(f"../../stimuli/originals/stimuli_{stim}.png", format="PNG")

    mask = Image.open(f"../../stimuli/transparent_mask/stimuli_{stim}.png")
    if mask.mode not in ("RGB", "RGBA"):
        mask = mask.convert("RGBA")
        mask.save(f"../../stimuli/transparent_mask/stimuli_{stim}.png", format="PNG")

# prompt="a basic object or figure, drawn with digital fineliner sketchpen and simple imperfect children's rough doodle on white background, monochrome",
prompts = [
            "creative minimalist black-on-white drawing, lineart-style on white background, drawn with digital fineliner, no color or shading",
            "creative minimalist black-on-white drawing of day-to-day object or scene, lineart-style on white background, drawn with digital fineliner, no color or shading",
            "creative minimalist black-on-white drawing of living figures (human, animal or creature), lineart-style on white background, drawn with digital fineliner, no color or shading"
            # "minimalist black-on-white drawing or simple doodle, drawn with digital fineliner on white background, creative",
            # "minimalist black-on-white drawing or simple doodle of an object or scene, drawn with digital fineliner on white background, creative",
            # "minimalist black-on-white drawing or simple doodle of living figures (human, animal or creature), drawn with digital fineliner on white background, creative"
        ]

maxnum = 10
runstart = 26
for ind, p in enumerate(prompts):
    for stim in ["I"]:
        
        output_dir = f"../../data/AI/prompt_{ind + 1}/stimuli_{stim}/"
        print(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        for run in range(5):

            response = client.images.edit(
                image=open(f"../../stimuli/originals/stimuli_{stim}.png", "rb"),
                mask=open(f"../../stimuli/padded_mask/stimuli_{stim}.png", "rb"),
                prompt=p,
                n=maxnum,
                size="512x512",
            )

            urls = [resp.url for resp in response.data]

            for i, url in enumerate(urls):
                response = requests.get(url)

                if response.status_code == 200:
                    # Open a file to write the image
                    # with open(output_dir + f"Dalle{(runstart + run) * maxnum + i + 1}.png", "wb") as file:
                    #     file.write(response.content)
                    image = Image.open(io.BytesIO(response.content))
                    resized_image = image.resize((400, 400))
                    resized_image.save(output_dir + f"Dalle{(runstart + run) * maxnum + i + 1}.png", format="PNG")

                else:
                    print(f"Failed to download image. Status code: {response.status_code}")