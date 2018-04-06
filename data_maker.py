
import comp_vis.img_tools as it
import comp_vis.live_tools as lt
import sys

def main():
    # Check that requisite arguments have been provided
    if len(sys.argv) < 4:
        sys.stderr.write("ERROR: Not all arguments provided. Should follow format:\n" +
                         "data_maker.py [camera_num (usually 0)] [bones_write_path] [rocks_write_path]")
        exit(1)

    camera_num  = int(sys.argv[1])
    bones_write_path = sys.argv[2]
    rocks_write_path = sys.argv[3]

    bone_images, rock_images = lt.data_gathering(camera_num)
    it.save_images(bone_images, bones_write_path)
    it.save_images(rock_images, rocks_write_path)

main()