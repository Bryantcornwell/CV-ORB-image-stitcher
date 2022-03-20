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
            old_x = int(old_coor[0] / old_coor[2])
            old_y = int(old_coor[1] / old_coor[2])
            # Update new image if old coordinate
            # is within the image coordinate bounds
            if old_y < img.shape[0] and \
                old_x < img.shape[1] and \
                old_y > 0 and \
                old_x > 0:

                    new_img[r, c] = img[old_y, old_x]

    return new_img




if __name__ == '__main__':

    # Store arguments in variables
    passed_n = int(sys.argv[1])
    first_img = sys.argv[2]
    second_img = sys.argv[3]
    output_img = sys.argv[4]

    # Store matching points in array
    matching_points = []
    i = 5
    while i < len(sys.argv):
        matching_points += [[
            [sys.argv[i], sys.argv[i+1], 1], 
            [sys.argv[i+2], sys.argv[i+3], 1]
        ]]
        i += 4

    matching_points = np.array(matching_points)


    # Get first image as array
    first_img = cv2.imread(first_img)

    # Apply transformation
    output = Apply_Transformation(first_img, passed_n)

    # Save output
    cv2.imwrite(output_img, output)

