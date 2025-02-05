import os
from openai import OpenAI
from PIL import Image
import requests
import io

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
            # "complete the image on the right in the same style as the one shown on the left, but make a different object."
            # "make a different object in the style of the left image"
            # "make a mushroom"
            "complete the triptych by drawing creative objects"
        ]

maxnum = 5
runstart = 3
for ind, p in enumerate(prompts):
    for stim in ["G"]:
        
        output_dir = "./"
        print(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        for run in range(1):

            response = client.images.edit(
                image=open(f"combined_image_GIR.png", "rb"),
                mask=open(f"combined_mask_IR.png", "rb"),
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
                    resized_image = image.resize((1200, 400))
                    resized_image.save(output_dir + f"Dalle{(runstart + run) * maxnum + i + 1}.png", format="PNG")
                else:
                    print(f"Failed to download image. Status code: {response.status_code}")