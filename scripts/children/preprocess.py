import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt


extra = 0
def crop_image(image, l):
    top, bottom, left, right = np.array(l) - extra
    height, width = image.shape[:2]
    toret = image[top:height - bottom, left:width - right]
    return toret
    
#top, botton, left, right
def find_crop_sizes(stim, inv, G_4010=False):
    if G_4010:
        return [29, 18, 122, 110]
    if stim == "G" and inv == False:
        return [38, 184, 110, 385]
    elif stim == "G" and inv == True:
        return [52, 200, 138, 378]
    elif stim == "I" and inv == False:
        return [48, 189, 115, 395]
    elif stim == "I" and inv == True:
        return [39, 192, 133, 373]
    elif stim == "R" and inv == True:
        return [52, 202, 117, 393]
    elif stim == "R" and inv == False:
        return [54, 198, 133, 377]

df = pd.read_csv("../../csvs/children_product.csv")

for index, row in df.iterrows():
    
    image_path = "../" + row["filepath"][:8] + "_" + row["filepath"][8:]
    im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Erode the image slightly to remove small noise
    small_kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, small_kernel, iterations=1)
    
    # Dilation to thicken the lines
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=3)
    image = cv2.addWeighted(image, 0.4, dilated, 0.6, 0)
    
    # Inverting the image back to get black lines on a white background
    image = cv2.bitwise_not(image)
    original_height, original_width = image.shape[:2]

    # Calculate the new width (reduce by ~3 times) and maintain aspect ratio
    new_width = int(original_width / 2.77)
    new_height = int(original_height * (new_width / original_width))
    image = cv2.resize(image, (new_width, new_height))
    
    # Crop image
    if image_path == "../../data/_children/stimuli_G/4010.png":
        image = crop_image(image, find_crop_sizes(row["stimuli"], row["inverted"], True))
    else:
        image = crop_image(image, find_crop_sizes(row["stimuli"], row["inverted"]))

    # Resize to 400 x 400
    image = cv2.resize(image, (400, 400))
    
    # Save the final image
    output_path = "../" + row["filepath"]
    cv2.imwrite(output_path, image)