#!/usr/local/bin/python3

import sys
import itertools


import cv2
import numpy as np


def euclidean_distance(point1, point2):
    distance = np.sqrt(np.sum(np.power(point1 - point2, 2)))
    return distance


def distance(point1, point2, kind='euclidean'):
    if kind == 'euclidean':
        return euclidean_distance(point1, point2)

def orb_sift_match(image_a, image_b):
    # Does some shit
    # Returns Boolean (True or False)
    # Possibly return distance to closest and second closest match too
    # E.g True/False
    # (True/False, distance, another_distance)

    #return '_3' in image_b or '_5' in image_b

    # img1 = cv2.imread(image_a, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(image_a)
    img2 = cv2.imread(image_b)

    # you can increase nfeatures to adjust how many features to detect
    orb = cv2.ORB_create(nfeatures=200)

    # detect features
    (keypoints1, descriptors1) = orb.detectAndCompute(img1, None)
    (keypoints2, descriptors2) = orb.detectAndCompute(img2, None)

    # Test image
    test_img = cv2.hconcat([img1, img2])
    draw = ImageDraw.Draw(test_img)
    #cv2.imshow('Horizontal', test_img)
    #cv2.waitKey(0)

    # Threshold
    threshold = 0.05
    # Two for loops to iterate through each feature point and calculate the Euclidean Distance
    for p1 in keypoints1:
        previous_distance = 10000
        point_list = []
        distance_list = []
        nearest_match = 0
        second_match = 0
        for p2 in keypoints2:
            #point_list.append(p2)
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
        if match < threshold:
            point_list.append([p1, nearest_match])
            draw.line((p1, (nearest_match[0]+img1.width, nearest_match[1])), fill=(255, 0, 0), width=7)
        #   Create a visual indication between the matched points for both images
        else:
            pass

    cv2.imshow('Horizontal', test_img)
    cv2.waitKey(0)
    #cv2.imwrite("lincoln-orb.jpg", img1)

def cluster_images(images):

    # Get all possible pairs of images
    pairings = [pair for pair in itertools.product(images, images) if pair[0] != pair[1] and pair[0] < pair[1]]
    # Determine whether they have an ORB/SIFT Match
    matches = {pair: orb_sift_match(pair[0], pair[1]) for pair in pairings}
    # Select only matched pairs
    matched_pairs = [str(pair) for pair in matches if matches[pair]]

    print('\n'.join(matched_pairs))

def main(images, output, k=2):

    orb_sift_match("part1-images/bigben_2.jpg", "part1-images/bigben_3.jpg")
    #cluster_images(images)


if __name__ == '__main__':
    # Step 1 Determine ORB Matching
    try:
        k = sys.argv[1]
        images = sys.argv[2:-1]
        output = sys.argv[-1]
    except:
        raise Exception(f'Usage: python3 part1.py <k>')

    main(images, output, k)
