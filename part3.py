from part1 import orb_sift_match


if __name__ == '__main__':

    try:
        _, image_1, image_2, output = sys.argv
    except:
        raise Exception(f'Usage: python3 part3.py <image_1> <image_2> <output_image>')