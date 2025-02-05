from PIL import Image, ImageDraw

image1 = Image.open("../../data/adults/stimuli_G/27__3565_9.jpg")  # Placeholder for image 1
image2 = Image.open("../../stimuli/originals/stimuli_R.png")  # Placeholder for image 2

combined_image = Image.new("RGB", (800, 400), color="white")
combined_image.paste(image1, (0, 0))  # Paste the first image
combined_image.paste(image2, (400, 0))  # Paste the second image

draw = ImageDraw.Draw(combined_image)
draw.line((400, 0, 400, 400), fill="black", width=1)  # Vertical line

combined_image.save("combined_image_GR.png")


image1 = Image.open("../../data/adults/stimuli_G/27__3565_9.jpg")  # Placeholder for image 1
image2 = Image.open("../../stimuli/originals/stimuli_I.png")  # Placeholder for image 2
image3 = Image.open("../../stimuli/originals/stimuli_R.png")  # Placeholder for image 2

combined_image = Image.new("RGB", (1200, 400), color="white")
combined_image.paste(image1, (0, 0))  # Paste the first image
combined_image.paste(image2, (400, 0))  # Paste the second image
combined_image.paste(image3, (800, 0))  # Paste the second image

draw = ImageDraw.Draw(combined_image)
draw.line((400, 0, 400, 400), fill="black", width=1)  # Vertical line
draw.line((800, 0, 800, 800), fill="black", width=1)  # Vertical line

combined_image.save("combined_image_GIR.png")