import img_tools as it
import sys


if len(sys.argv) < 2:
    print("ERROR: At least one command line argument required.")
    print("\t(i.e. ../sample_images/multi_images)")
    exit()

path = sys.argv[1]
images = it.load_images(path)
images = it.generate_images_from_multi(images[0])
for img in images:
    it.show(img)