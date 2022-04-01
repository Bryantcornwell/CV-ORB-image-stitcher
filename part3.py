import sys
from pathlib import Path

import numpy as np

from part1 import orb_sift_match

def ransac(point_matches, )

def main(image_1, image_2, output):

    point_matches = np.array(orb_sift_match(image_1, image_2))
    point_matches = point_matches[:,-2:]
    point_matches = np.array(list(map(list, point_matches)))
    return

if __name__ == '__main__':

    try:
        image_1, image_2, output = sys.argv[1:]
    except:
        raise Exception(f'Usage: python3 part3.py <image_1> <image_2> <output_image>')

    main(Path(image_1), Path(image_2), Path(output))
