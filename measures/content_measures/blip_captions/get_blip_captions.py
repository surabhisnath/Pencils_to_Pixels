from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
from tqdm import tqdm
import os

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

prompts = ["1", "2", "3"]
for prompt in prompts:
    creator = f"AI_prompt{prompt}"
    for stimulus in ["G", "I", "R"]:

        image_folder = f"../../../data/AI/prompt_{prompt}/stimuli_{stimulus}/"  # Replace with the path to your local image
        
        rows = []
        for imagename in tqdm(sorted(os.listdir(image_folder))):
            raw_image = Image.open(image_folder + imagename).convert("L")

            # text = "a drawing of"
            # inputs = processor(raw_image, text, return_tensors="pt").to("cuda")
            # out = model.generate(**inputs, max_new_tokens=50)
            # caption = processor.decode(out[0], skip_special_tokens=True)
            # print(caption)

            # unconditional image captioning
            inputs = processor(raw_image, return_tensors="pt").to("cuda")
            out = model.generate(**inputs)

            rows.append({"creator": creator, "stimuli": stimulus, "image":imagename, "filepath": image_folder[image_folder.find("../data/"):] + imagename, "blip_caption": processor.decode(out[0], skip_special_tokens=True)})
            
        df = pd.DataFrame(rows)
        df.to_csv(f"output/{creator}_{stimulus}.csv", index=False)