# https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
import os
import base64
import pandas as pd
from tqdm import tqdm 
from openai import OpenAI
OpenAI.api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI()
from PIL import Image, ImageDraw
import io

# Function to encode the image
# def encode_image(image_path):
#   with open(image_path, "rb") as image_file:
#     return base64.b64encode(image_file.read()).decode('utf-8')

creators = ["adults", "AI"]

for creator in tqdm(creators):
    rows = []
    df = pd.read_csv(f"../../../csvs/{creator}_product.csv")
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing rows for {creator}"):
        stim = row["stimuli"]
        image1 = Image.open(f"../../../stimuli/originals/stimuli_{stim}.png")
        image2 = Image.open(f"../../{row["filepath"]}")
        combined_image = Image.new("RGB", (800, 400), color="white")
        combined_image.paste(image1, (0, 0))  # Paste the first image
        combined_image.paste(image2, (400, 0))  # Paste the second image
        draw = ImageDraw.Draw(combined_image)
        combined_image.save("combined_image.png")    

        buffer = io.BytesIO()
        combined_image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": "The left image was converted into the right image. Rate the creativity on a scale of 1 to 10 in {}. Give a short explanation.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url":  f"data:image/jpeg;base64,{base64_image}"
                    },
                },
                ],
            }
            ],
        )
      
        rows.append({"creator":creator, "stimuli":stim, "filepath": row["filepath"], "gpt_creativity": response.choices[0].message.content})
    print(rows)
    df = pd.DataFrame(rows)
    df.to_csv(f"output/{creator}_{stim}.csv", index=False)