import cv2
import numpy as np
import sys


def euclidean_distance(point1, point2):
    distance= np.sqrt(np.sum(np.power(point1 - point2, 2)))
    return distance


def distance(point1, point2, kind='euclidean'):
    if kind == 'euclidean':
        return euclidean_distance(point1, point2)


def main(image_a, image_b, k=2):

    img1 = cv2.imread("part1-images/bigben_2.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("part1-images/bigben_3.jpg", cv2.IMREAD_GRAYSCALE)

    # you can increase nfeatures to adjust how many features to detect
    orb = cv2.ORB_create(nfeatures=1000)

    # detect features
    (keypoints1, descriptors1) = orb.detectAndCompute(img1, None)
    (keypoints2, descriptors2) = orb.detectAndCompute(img2, None)

    # Test image
    test_img = cv2.hconcat([img1, img2])
    cv2.imshow('Horizontal', test_img)
    cv2.waitKey(0)

    # Threshold

    # Two for loops to iterate through each feature point and calculate the Euclidean Distance
    """
    for p1 in keypoints1:
        previous_distance = 10000
        point_list = []
        distance_list = []
        nearest_match = 0
        second_match = 0
        for p2 in keypoints2:
            point_list.append(p2)
            point_distance = distance(p1, p2)
            distance_list.append(point_distance)
            if point_distance < previous_distance:
                second_match = nearest_match
                nearest_match = p2
            previous_distance = point_distance
            # Need to determine if we want a quicker computational approach

        # feature match
        distance_closest = p1 - nearest_match
        distance_2ndclosest = p1 - second_match
        match = distance_closest / distance_2ndclosest
        # if match < threshold then it is a match
        #   Create a visual indication between the matched points for both images
        # else:pass
    """
    #cv2.imshow('Horizontal', test_img)
    #cv2.imwrite("lincoln-orb.jpg", img1)


if __name__ == '__main__':
    # Step 1 Determine ORB Matching
    try:
        k = sys.argv[1]
        images = sys.argv[2:-1]
        output = sys.argv[-1]
    except:
        raise Exception(f'Usage: python3 part1.py <k>')

    main(k)
