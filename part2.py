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
    transform_array = TRANSFORMATIONS[transform_idx]

    # Initialize new image
    new_img = np.zeros(shape=img.shape)

    # Loop through coordinates and apply transformation
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            current_coor = np.array([c, r, 1])
            new_coor = np.matmul(transform_array, current_coor)
            new_x = int(new_coor[0] / new_coor[2])
            new_y = int(new_coor[1] / new_coor[2])
            # Update new image if new coordinate
            # is within the image coordinate bounds
            if new_y < img.shape[0] and \
                new_x < img.shape[1] and \
                new_y > 0 and \
                new_x > 0:

                    new_img[new_y, new_x] = img[r, c]

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

