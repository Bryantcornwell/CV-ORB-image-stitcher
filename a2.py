#!/usr/bin/env python3

import sys
from part1 import main as part1main
from part2 import main as part2main
from part3 import main as part3main

if __name__ == '__main__':

    part_n = sys.argv[1]

    if part_n == 'part1':

        try:
            k = int(sys.argv[2])
            images = sys.argv[3:-1]
            output = sys.argv[-1]
        except:
            raise Exception(f'Usage: python3 part1.py <k> <img_1> <img_2> ... <img_n> <output_file>')

        part1main(images, output, k)

    elif part_n == 'part2':

        try:
            passed_n = int(sys.argv[2])
            first_img = sys.argv[3]
            second_img = sys.argv[4]
            output_img = sys.argv[5]
            passed_coordinates = sys.argv[6:]
        except:
            raise Exception(f'Usage: python3 part2.py <n> <img1> <img2> <outImg> <x11,y11> <x12,y12> ... <x1n,y1n> <x21,y21> <x22,y22> ... <x2n,y2n>')

        part2main(passed_n, first_img, second_img, output_img, passed_coordinates)

    elif part_n == 'part3':

        try:
            image_1, image_2, output = sys.argv[1:]
        except:
            raise Exception(f'Usage: python3 part3.py <image_1> <image_2> <output_image>')

        part3main(Path(image_1), Path(image_2), Path(output))
