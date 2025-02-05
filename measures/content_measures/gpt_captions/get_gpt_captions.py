# https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
import os
import base64
import pandas as pd

from openai import OpenAI
OpenAI.api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


prompts = ["1", "2", "3"]
for prompt in prompts:
  creator = f"AI_prompt{prompt}"

  for stimulus in ["G", "I", "R"]:
    image_folder = f"../../../data/AI/prompt_{prompt}/stimuli_{stimulus}/"

    rows = []
    for image in sorted(os.listdir(image_folder)):
      print(image)
      if image[-4:] not in [".png", ".jpg", "jpeg"]:
        continue

      # Getting the base64 string
      base64_image = encode_image(image_folder + image)

      response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": "Describe the content (not style) of the drawing. Use maximum 15 words. If it is hard to interpret, output 'hard to interpret'. In the end, give a caption confidence score (1-5) in '[]'.",
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
      
      rows.append({"creator":creator, "stimuli":stimulus, "filepath": image_folder[image_folder.find("../data/"):] + image, "gpt_caption":response.choices[0].message.content})

    df = pd.DataFrame(rows)
    df.to_csv(f"output/{creator}_{stimulus}.csv", index=False)