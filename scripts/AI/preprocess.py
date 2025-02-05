import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from PIL import Image

for prompt in ["1", "2", "3"]:
    for stimulus in ["G", "I", "R"]:
        print(prompt, stimulus)
        image_folder = f"../../data/_AI/prompt_{prompt}/stimuli_{stimulus}/"
        output_folder = f"../../data/AI/prompt_{prompt}/stimuli_{stimulus}/"
        os.makedirs(output_folder, exist_ok=True)

        for imagename in tqdm(sorted(os.listdir(image_folder))):
            image = cv2.imread(image_folder + imagename, cv2.IMREAD_GRAYSCALE)
            image = cv2.bitwise_not(image)
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(image, kernel, iterations=1)
            image = cv2.addWeighted(image, 0.5, dilated, 0.5, 0)
            image = cv2.bitwise_not(image)
            cv2.imwrite(output_folder + imagename, image)