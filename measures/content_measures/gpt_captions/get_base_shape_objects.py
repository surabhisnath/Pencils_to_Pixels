import os
import base64
from openai import OpenAI
OpenAI.api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image(f"../../../stimuli/originals/stimuli_R.png")

response = client.chat.completions.create(
model="gpt-4o",
messages=[
    {
    "role": "user",
    "content": [
        {
        "type": "text",
        "text": "List 10 concepts easiest to draw to complete this image."
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
temperature=0
)

print(response.choices[0].message.content)


# G
# 1. Umbrella
# 2. Mushroom
# 3. Jellyfish
# 4. Parachute
# 5. Hot air balloon
# 6. Igloo
# 7. Dome tent
# 8. UFO
# 9. Bell
# 10. Helmet

# 1. Umbrella
# 2. Mushroom
# 3. Jellyfish
# 4. Hot air balloon
# 5. Parachute
# 6. Igloo
# 7. Dome tent
# 8. Turtle
# 9. Bell
# 10. UFO

# 1. Umbrella
# 2. Mushroom
# 3. Jellyfish
# 4. Hot air balloon
# 5. Parachute
# 6. Igloo
# 7. Dome tent
# 8. Bell
# 9. UFO
# 10. Helmet

# 1. Umbrella
# 2. Mushroom
# 3. Jellyfish
# 4. Parachute
# 5. Hot air balloon
# 6. Igloo
# 7. Dome tent
# 8. Turtle
# 9. UFO
# 10. Bell

# 1. Umbrella
# 2. Mushroom
# 3. Jellyfish
# 4. Hot air balloon
# 5. Parachute
# 6. Igloo
# 7. Dome tent
# 8. Bell
# 9. UFO
# 10. Helmet

# X:
# 1. Umbrella
# 2. Mushroom
# 3. Jellyfish
# 4. Hot air balloon
# 5. Igloo



# I
# 1. Padlock
# 2. Doorway
# 3. Tombstone
# 4. Archway
# 5. Backpack
# 6. Robot head
# 7. Castle tower
# 8. Bell
# 9. Lantern
# 10. Igloo

# 1. **Doorway**: Add a door with a handle.
# 2. **Archway**: Complete the arch and add some bricks or stones.
# 3. **Tombstone**: Add inscriptions or decorative elements.
# 4. **Window**: Draw window panes and a sill.
# 5. **Bridge**: Extend the lines to form a simple bridge.
# 6. **Tunnel Entrance**: Add details like bricks or a path leading in.
# 7. **Castle Gate**: Add a portcullis or decorative elements.
# 8. **Picture Frame**: Add a picture or design inside.
# 9. **Mirror**: Add reflective details and a frame.
# 10. **Book Cover**: Add a title and some decorative elements.

# 1. Padlock
# 2. Doorway
# 3. Archway
# 4. Tombstone
# 5. Backpack
# 6. Robot head
# 7. Castle tower
# 8. Bell
# 9. Lantern
# 10. Igloo

# 1. **Doorway**: Add a door with a handle.
# 2. **Archway**: Complete the arch and add bricks or stones.
# 3. **Window**: Add glass panes and a window sill.
# 4. **Tombstone**: Add an inscription or decorative elements.
# 5. **Bridge**: Extend the lines to create a simple bridge.
# 6. **Tunnel Entrance**: Add depth and shading to suggest a tunnel.
# 7. **Castle Gate**: Add a portcullis and castle walls.
# 8. **Picture Frame**: Add a border and a picture inside.
# 9. **Book Cover**: Add a spine and title.
# 10. **Mirror**: Add a reflective surface and a frame.

# 1. **Padlock**: Add a keyhole and a shackle to make it look like a padlock.
# 2. **Doorway**: Extend the lines to form a door and add details like a doorknob.
# 3. **Archway**: Complete the arch and add bricks or stones for texture.
# 4. **Tombstone**: Add inscriptions or decorative elements.
# 5. **Robot**: Add a face and body below the arch to create a robot.
# 6. **Backpack**: Add straps and pockets to form a backpack.
# 7. **Castle Gate**: Add a portcullis and stone details.
# 8. **Window**: Add panes and curtains to make it a window.
# 9. **Bridge**: Extend the lines to form a bridge with supports.
# 10. **Tunnel Entrance**: Add details like tracks or a road leading into it.

# X:
# Door
# Arch
# Window
# Tombstone
# Padlock


# # R
# 1. House
# 2. Arrow
# 3. Diamond
# 4. Shield
# 5. Kite
# 6. Envelope
# 7. Heart
# 8. Ice cream cone
# 9. Tree
# 10. Rocket


# X:
# House
# arrow
# shield
# diamond
# kite