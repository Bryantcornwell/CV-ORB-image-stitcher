



import cv2
import numpy as np
import sys


def eu_dist(point1, point2):
    dist = np.sqrt(np.sum(np.power(point1 - point2, 2)))
    return dist


# Step 1 Determine ORB Matching
k = sys.argv[1]

img1 = cv2.imread("part1-images/bigben_2.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("part1-images/bigben_3.jpg", cv2.IMREAD_GRAYSCALE)

# you can increase nfeatures to adjust how many features to detect
orb = cv2.ORB_create(nfeatures=1000)

# detect features
(keypoints1, descriptors1) = orb.detectAndCompute(img1, None)
(keypoints2, descriptors2) = orb.detectAndCompute(img2, None)

# Threshold

# Two for loops to iterate through each feature point and calculate the Euclidean Distance

for p1 in keypoints1:
    prev_dist = 10000
    point_list = []
    dist_list = []
    nearest_match = 0
    second_match = 0
    for p2 in keypoints2:
        point_list.append(p2)
        point_dist = eu_dist(p1, p2)
        dist_list.append(point_dist)
        if prev_dist < point_dist:
            second_match = nearest_match
            nearest_match = p2
        prev_dist = point_dist
        # Need to determine if we want a quicker computational approach

    # feature match
    dist_closest = p1 - nearest_match
    dist_2ndclosest = p1 - second_match
    match = dist_closest / dist_2ndclosest
    # if match < threshold then it is a match
    #   Create a visual indication between the matched points for both images
    # else:pass

"""
# put a little X on each feature
for i in range(0, len(keypoints)):
    print("Keypoint #%d: x=%d, y=%d, descriptor=%s, distance between this descriptor and descriptor #0 is %d" % (
    i, keypoints[i].pt[0], keypoints[i].pt[1], np.array2string(descriptors[i]),
    cv2.norm(descriptors[0], descriptors[i], cv2.NORM_HAMMING)))
    for j in range(-5, 5):
        img[int(keypoints[i].pt[1]) + j, int(keypoints[i].pt[0]) + j] = 0
        img[int(keypoints[i].pt[1]) - j, int(keypoints[i].pt[0]) + j] = 255
"""
#cv2.imwrite("lincoln-orb.jpg", img1)