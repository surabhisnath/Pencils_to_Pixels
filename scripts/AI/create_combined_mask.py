from PIL import Image, ImageDraw

white_image = Image.new("RGBA", (400, 400), color=(255, 255, 255, 255))

mask_G = Image.open("../../stimuli/padded_mask/stimuli_G.png").convert("RGBA")
mask_I = Image.open("../../stimuli/padded_mask/stimuli_I.png").convert("RGBA")
mask_R = Image.open("../../stimuli/padded_mask/stimuli_R.png").convert("RGBA")

combined_image = Image.new("RGBA", (800, 400), color=(0, 0, 0, 0))
combined_image.paste(white_image, (0, 0))
combined_image.paste(mask_G, (400, 0))
combined_image.save("combined_mask_G.png")

combined_image = Image.new("RGBA", (800, 400), color=(0, 0, 0, 0))
combined_image.paste(white_image, (0, 0))
combined_image.paste(mask_I, (400, 0))
combined_image.save("combined_mask_I.png")

combined_image = Image.new("RGBA", (800, 400), color=(0, 0, 0, 0))
combined_image.paste(white_image, (0, 0))
combined_image.paste(mask_R, (400, 0))
combined_image.save("combined_mask_R.png")

combined_image = Image.new("RGBA", (1200, 400), color=(0, 0, 0, 0))
combined_image.paste(white_image, (0, 0))
combined_image.paste(mask_I, (400, 0))
combined_image.paste(mask_R, (800, 0))
combined_image.save("combined_mask_IR.png")