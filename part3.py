import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
import cv2

from part1 import orb_sift_match, pad_image
from part2 import get_projection_matrix, test_transition_matrix, apply_transformation

def ransac(point_matches, sample_size, iterations, threshold, min_match_sample_size):
    # RANSAC some $#17

    best_transform = None
    best_sample = None
    best_error = np.inf

    print(len(point_matches))
    i = 0
    while i < iterations:
        sample_space = deepcopy(point_matches)
        np.random.shuffle(sample_space)
        sample, sample_test = sample_space[:sample_size], sample_space[sample_size:]
        try:
            transform = get_projection_matrix(sample)
        except np.linalg.LinAlgError as err:
            continue
        sample_test = np.array([sample_test[j] for j in range(sample_test.shape[0]) if test_transition_matrix(transform, [sample_test[j]]) < threshold])
        if len(sample_test) > min_match_sample_size or best_transform is None:
            if best_transform is None:
                new_sample = sample_space
            else:
                new_sample = np.concatenate([sample, sample_test])
            error = test_transition_matrix(transform, new_sample)
            if error < best_error:
                print(i, error)
                best_transform = transform
                best_error = error
                best_sample = new_sample

        i += 1

    print('Best sample error:', test_transition_matrix(best_transform, best_sample))
    return best_transform, best_sample

def main(image_1, image_2, output):

    # CALCULATE CENTROID
    # MATCH ON CENTROID
    # AVERAGE PIXEL BY PIXEL
    # SMOOTH IMAGE
    # FIGURE OUT HOW TO GET THE IMAGES TO MATCH IN SIZE WITHOUT FUCKING SHIT UP!

    image_a = cv2.imread(str(Path(image_1)))
    image_b = cv2.imread(str(Path(image_2)))

    point_matches = np.array(orb_sift_match(image_1, image_2))
    point_matches = point_matches[:,-2:]
    point_matches = np.array(list(map(list, point_matches)))
    transform_matrix, shared_coordinates = ransac(point_matches, 4, len(point_matches) ** 3, 0.75, int(0.1*len(point_matches)))
    padded_image = pad_image(image_b, 50, 50, 50, 50)
    cv2.imwrite('padded_image.png', padded_image)
    transformed = apply_transformation(padded_image, transform_matrix)
    cv2.imwrite(str(Path(output)), transformed)
    

if __name__ == '__main__':

    try:
        image_1, image_2, output = sys.argv[1:]
    except:
        raise Exception(f'Usage: python3 part3.py <image_1> <image_2> <output_image>')

    main(Path(image_1), Path(image_2), Path(output))
