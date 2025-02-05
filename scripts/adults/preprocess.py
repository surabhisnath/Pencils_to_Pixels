import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from PIL import Image


for stimulus in ["G", "I", "R"]:
    print(stimulus)
    image_folder = f"../../data/_adults/stimuli_{stimulus}/"
    output_folder = f"../../data/adults/stimuli_{stimulus}/"
    os.makedirs(output_folder, exist_ok=True)

    for imagename in tqdm(sorted(os.listdir(image_folder))):
        image = cv2.imread(f"../../data/adults/stimuli_{stimulus}/" + imagename, cv2.IMREAD_GRAYSCALE)
        image = cv2.bitwise_not(image)
        kernel = np.ones((2,2), np.uint8)  # Adjust kernel size for more/less erosion
        eroded = cv2.erode(image, kernel, iterations=1)
        image = cv2.addWeighted(image, 0.5, eroded, 0.5, 0)
        image = cv2.bitwise_not(image)
        plt.imsave("eroded_image.png", image, cmap="gray")

        # image = cv2.imread(image_folder + imagename, cv2.IMREAD_GRAYSCALE)
        # _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
        # kernel = np.ones((2,2), np.uint8)
        # image = cv2.dilate(image, kernel, iterations=1)
        # image = cv2.bitwise_not(image)
        cv2.imwrite(output_folder + imagename, image)