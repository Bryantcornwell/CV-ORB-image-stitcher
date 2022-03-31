#!/usr/local/bin/python3

import sys
import cv2
import numpy as np



TRANSFORMATIONS = [

    # Apply Test Transform to Lincoln
    np.array([
        [0.907, 0.258, -182],
        [-0.153, 1.44, 58],
        [-0.000306, 0.000731, 1]
    ]),

    # Apply a Simple Translation
    np.array([
        [1, 0, 100],
        [0, 1, 200],
        [0, 0, 1]
    ])

]



def Apply_Transformation(img, transform_idx):

    # Get transformation to apply
    # Take the inverse to perform inverse warping
    transform_array = np.linalg.inv(TRANSFORMATIONS[transform_idx])

    # Initialize new image
    new_img = np.zeros(shape=img.shape)

    # Loop through coordinates and
    # apply inverse transformation
    for r in range(new_img.shape[0]):
        for c in range(new_img.shape[1]):
            current_coor = np.array([c, r, 1])
            old_coor = np.matmul(transform_array, current_coor)
            old_x = old_coor[0] / old_coor[2]
            old_y = old_coor[1] / old_coor[2]
            # Update new image if old coordinate
            # is within the image coordinate bounds
            if old_y < img.shape[0] and \
                old_x < img.shape[1] and \
                old_y > 0 and \
                old_x > 0:

                    new_img[r, c] = Apply_Interpolation(old_x, old_y, img)

    return new_img


def Apply_Interpolation(x, y, img):

    x_ceiling = np.ceil(x)
    y_ceiling = np.ceil(y)

    x_remainder = x % 1
    y_remainder = y % 1

    # Apply Bilinear Interpolation
    if x % 1 > 0 and y % 1 > 0 and \
        x_ceiling < img.shape[1] and y_ceiling < img.shape[0]:

            top_left = img[int(y_ceiling-1), int(x_ceiling-1)]
            top_right = img[int(y_ceiling-1), int(x_ceiling)]
            bot_left = img[int(y_ceiling), int(x_ceiling-1)]
            bot_right = img[int(y_ceiling), int(x_ceiling)]

            top_left_weight = (1 - x_remainder) * (1 - y_remainder)
            top_right_weight = (x_remainder) * (1 - y_remainder)
            bot_left_weight = (1 - x_remainder) * (y_remainder)
            bot_right_weight = (x_remainder) * (y_remainder)

            return top_left * top_left_weight + \
                top_right * top_right_weight + \
                bot_left * bot_left_weight + \
                bot_right * bot_right_weight

    # Apply Horizontal Linear Interpolation
    elif x % 1 > 0 and y % 1 == 0 and x_ceiling < img.shape[1]:

        left = img[int(y), int(x_ceiling - 1)]
        right = img[int(y), int(x_ceiling)]

        left_weight = 1 - x_remainder
        right_weight = x_remainder

        return left * left_weight + \
            right * right_weight

    # Apply Vertical Linear Interpolation
    elif x % 1 == 0 and y % 1 > 0 and y_ceiling < img.shape[0]:

        top = img[int(y_ceiling - 1), int(x)]
        bot = img[int(y_ceiling), int(x)]

        top_weight = 1 - y_remainder
        bot_weight = y_remainder

        return top * top_weight + \
            bot * bot_weight

    # Interpolation not necessary
    else: return img[int(y), int(x)]


def Build_Matching_Coordinates(dof):
    matching_points = []

    i = 5
    while i < len(sys.argv):
        matching_points += [[
            # Need to formation matching point arrays for linalg.solve
            # Degrees of freedom effect homogenous coordinate size?
            [sys.argv[i], sys.argv[i+1], 1], 
            [sys.argv[i+2], sys.argv[i+3], 1]
        ]]
        i += 4

    return np.array(matching_points)






if __name__ == '__main__':

    # Store arguments in variables
    passed_n = int(sys.argv[1])
    first_img = sys.argv[2]
    second_img = sys.argv[3]
    output_img = sys.argv[4]
    points = [map(int, point.split(',')) for point in sys.argv[5:]]

    matching_coordinates = np.array()
    degrees_of_freedom = 0

    # If passed_n = 1 (Translation)
    # 2 degrees of freedom
    # Need 1 coordinate
    if passed_n == 1:
        if len(sys.argv) < 7:
            print('Please include one coordinate match for translation transformation')
        else:
            matching_coordinates = Build_Matching_Coordinates(2)
            degrees_of_freedom = 2

    # If passed_n = 2 (Euclidean)
    # 3 degrees of freedom
    # Need 2 coordinates
    elif passed_n == 2:
        if len(sys.argv) < 11:
            print('Please include two coordinate matches for Euclidean transformation')
        else:
            matching_coordinates = Build_Matching_Coordinates(3)
            degrees_of_freedom = 3

    # If passed_n = 3 (Affine)
    # 6 degrees of freedom
    # Need 3 coordinates
    elif passed_n == 3:
        if len(sys.argv) < 15:
            print('Please include three coordinate matches for Affine transformation')
        else:
            matching_coordinates = Build_Matching_Coordinates(6)
            degrees_of_freedom = 6

    # If passed_n = 4 (Projective)
    # 8 degrees of freedom
    # Need 4 coordinates
    elif passed_n == 4:
        if len(sys.argv) < 19:
            print('Please include four coordinate matches for Projection transformation')
        else:
            matching_coordinates = Build_Matching_Coordinates(8)
            degrees_of_freedom = 8

    # Invalid n value
    else:
        print('Please enter acceptable n value')
        print('1 = Translation\n2 = Euclidean\n3 = Affine\n4 = Projective')


    # If coordinates not empty, apply transformation
    if matching_coordinates.shape[0] > 0:

        # Get translation matrix
        # translation_matrix = Build_Translation_Matrix(matching_coordinates, degrees_of_freedom)

        # Get first image as array
        first_img = cv2.imread(first_img)

        # Apply transformation
        output = Apply_Transformation(first_img, passed_n)

        # Save output
        cv2.imwrite(output_img, output)

