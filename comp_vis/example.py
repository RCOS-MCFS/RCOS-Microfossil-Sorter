import numpy as np
import img_tools as it
import sys


if len(sys.argv) < 2:
    print("ERROR: At least one command line argument required.")
    print("\t(i.e. ../sample_images/multi_images/bones)")
    exit()

path = sys.argv[1]
images = it.load_images(path)
images = it.generate_cropped_from_multi(images[0])

average_colors = []
for image in images:
    average_colors.append(it.average_color(image))

print(average_colors)