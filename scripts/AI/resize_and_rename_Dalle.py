import os
from PIL import Image

# Input and output directories

prompts = ["1", "2", "3"]
stimuli = ["G", "I", "R"]


for prompt in prompts:
    for stim in stimuli:
        input_dir = f"../../data/AI/prompt_{prompt}/stimuli_{stim}/"
        output_dir = f"../../data/AI/prompt_{prompt}/stimuli_{stim}_resized/"

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        num = 1
        for file_name in sorted(os.listdir(input_dir)):
            if file_name.endswith((".png", ".jpg", ".jpeg")):  # Process only image files
                # Open the image
                input_path = os.path.join(input_dir, file_name)
                image = Image.open(input_path)
                
                # Resize to 400x400 pixels
                resized_image = image.resize((400, 400), Image.Resampling.LANCZOS)
                
                # Save to the output directory
                output_path = os.path.join(output_dir, f"Dalle{str(num).zfill(2)}.png")
                resized_image.save(output_path)
                num+=1

        print(f"All images have been resized and saved in {output_dir}")