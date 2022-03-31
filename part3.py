import sys

from part1 import orb_sift_match


if __name__ == '__main__':

    try:
        image_1, image_2, output = sys.argv[1:]
    except:
        raise Exception(f'Usage: python3 part3.py <image_1> <image_2> <output_image>')

    print(image_1, image_2, output)
