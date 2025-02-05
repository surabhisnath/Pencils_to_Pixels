import os
from PIL import Image

pad_size = 2
directory = "../stimuli/transparent_mask/"
for filename in os.listdir(directory):
    if "png" in filename:
        f = os.path.join(directory, filename)
        img = Image.open(f)
        img = img.convert("RGBA")
        datas = img.getdata()
        newData = [0] * len(datas)
        for j, pixel in enumerate(datas):
            # if alpha is 255/opaque, we add the opaque padding
            if j > pad_size and j < (len(datas) - pad_size):
                # if its not at the beginning or end of the image
                if pixel[3] > 0 and datas[j - 1][3] == 0:
                    # if border: dark pixel, but transparent before
                    newData[j - pad_size : j] = [
                        (255, 255, 255, 255) for p in newData[j - pad_size : j]
                    ]
                    # all pixels until dark one
                    newData[j] = pixel  # datas[j]
                elif (
                    pixel[3] > 0 and datas[j + 1][3] == 0
                ):  # if border: transparent after
                    newData[j + 1 : j + pad_size + 1] = [
                        (255, 255, 255, 255) for p in newData[j + 1 : j + pad_size + 1]
                    ]
                    newData[j] = pixel  # datas[j]
                elif pixel[3] > 0:
                    newData[j] = pixel

                if (
                    pixel[3] > 0 and datas[j - img.size[0]][3] == 0
                ):  # if border and up theres nothing
                    up_pixels = list(
                        range(j - (pad_size * img.size[0]), j, img.size[0])
                    )
                    up_pixels.append(j - 1)
                    newData[j] = pixel
                    for u in up_pixels:
                        newData[u] = (255, 255, 255, 255)
                elif (
                    pixel[3] > 0 and datas[j + img.size[0]][3] == 0
                ):  # if border and down theres nothing
                    down_pixels = list(
                        range(j + 1, j + (pad_size * img.size[0]) + 1, img.size[0])
                    )
                    newData[j] = pixel
                    for d in down_pixels:
                        newData[d] = (255, 255, 255, 255)

        for i in range(0, len(newData)):
            # the pixels not included, we set to transparent
            if newData[i] == 0:
                newData[i] = (255, 255, 255, 0)

        img.putdata(newData)
        saved_dir = f"../stimuli/padded_mask/{filename}"
        img.save(saved_dir, "PNG")


# pad_size = 2
# directory = "../stimuli/originals/"
# for filename in os.listdir(directory):
#     if "png" in filename:
#         f = os.path.join(directory, filename)
#         img = Image.open(f)
#         img = img.convert("RGBA")
#         datas = img.getdata()
#         newData1 = [0] * len(datas)
#         newData = [0] * len(datas)
#         for j, pixel in enumerate(datas):
#             if pixel[0] != 0 or pixel[1] != 0 or pixel[2] != 0:
#                 newData1[j] = (pixel[0], pixel[1], pixel[2], 0)
#             else:
#                 newData1[j] = pixel

#         for j, pixel in enumerate(newData1):
#             # if alpha is 255/opaque, we add the opaque padding
#             if j > pad_size and j < (len(newData1) - pad_size):
#                 # if its not at the beginning or end of the image
#                 if pixel[3] > 0 and newData1[j - 1][3] == 0:
#                     # if border: dark pixel, but transparent before
#                     newData[j - pad_size : j] = [
#                         (255, 255, 255, 255) for p in newData[j - pad_size : j]
#                     ]
#                     # all pixels until dark one
#                     newData[j] = pixel  # newData1[j]
#                 elif (
#                     pixel[3] > 0 and newData1[j + 1][3] == 0
#                 ):  # if border: transparent after
#                     newData[j + 1 : j + pad_size + 1] = [
#                         (255, 255, 255, 255) for p in newData[j + 1 : j + pad_size + 1]
#                     ]
#                     newData[j] = pixel  # newData1[j]
#                 elif pixel[3] > 0:
#                     newData[j] = pixel

#                 if (
#                     pixel[3] > 0 and newData1[j - img.size[0]][3] == 0
#                 ):  # if border and up theres nothing
#                     up_pixels = list(
#                         range(j - (pad_size * img.size[0]), j, img.size[0])
#                     )
#                     up_pixels.append(j - 1)
#                     newData[j] = pixel
#                     for u in up_pixels:
#                         newData[u] = (255, 255, 255, 255)
#                 elif (
#                     pixel[3] > 0 and newData1[j + img.size[0]][3] == 0
#                 ):  # if border and down theres nothing
#                     down_pixels = list(
#                         range(j + 1, j + (pad_size * img.size[0]) + 1, img.size[0])
#                     )
#                     newData[j] = pixel
#                     for d in down_pixels:
#                         newData[d] = (255, 255, 255, 255)

#         for i in range(0, len(newData)):
#             # the pixels not included, we set to transparent
#             if newData[i] == 0:
#                 newData[i] = (255, 255, 255, 0)

#         img.putdata(newData)
#         saved_dir = f"../stimuli/padded_mask/{filename}"
#         img.save(saved_dir, "PNG")
