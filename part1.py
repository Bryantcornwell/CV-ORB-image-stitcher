#!/usr/local/bin/python3

import sys
import itertools
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, freeze_support
from functools import partial

from PIL import ImageDraw, Image, ImageOps
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm, trange


SAVE_IMG = False
runtime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def euclidean_distance(point1, point2):
    #print(type(cv2.KeyPoint_convert(keypoints=point1), type(point2)))
    #distance = np.sqrt(np.sum(np.power(point1 - point2, 2)))
    distance = np.linalg.norm(point1 - point2)
    return distance

def distance(point1, point2, kind='hamming'):
    if kind == 'euclidean':
        return euclidean_distance(point1, point2)
    elif kind == 'hamming':
        return hamming_distance(point1, point2)

def pad_image(image, top, bottom, left, right, color=(0,0,0)):

    Image.fromarray(image).save('test.png')
    cv2.imwrite('test2.png', image)

    width_padded = image.shape[1] + left + right
    height_padded = image.shape[0] + top + bottom
    padded_image = Image.new('RGB', (width_padded, height_padded), color)
    Image.fromarray(image).save('test.png')
    return np.asarray(padded_image.paste(Image.fromarray(image), (left, top)))

def old_match_points_code(descs1, descs2):
    
    # Two for loops to iterate through each feature point and calculate the Euclidean Distance
    point_list = []
    match_list = []
    distance_list = []
    #for p1, desc1 in tqdm(zip(keypoints1, descriptors1), desc='p1', position=2, leave=False, total=len(keypoints1), unit='points'):

    #paired_points = [(kp1, kp2) for kp1 in keypoints1 for kp2 in keypoints2]
    #paired_descriptors = [(desc1, desc2)]

    for p1, desc1 in zip(keypoints1, descriptors1):
        nearest_distance = np.inf
        second_distance = np.inf
        nearest_match_point = None
        nearest_match_desc = None
        second_match_point = None
        second_match_desc = None
        #for p2, desc2 in tqdm(zip(keypoints2, descriptors2), desc='p2', position=3, leave=False, total=len(keypoints2), unit='points'):
        for p2, desc2 in zip(keypoints2, descriptors2):
            #point_list.append(p2)
            point_distance = distance(desc1, desc2)
            if nearest_match_point is None:
                nearest_match_point = p2
                nearest_distance = point_distance
                nearest_match_desc = desc2
                continue
            elif second_match_point is None and point_distance > nearest_distance:
                second_match_point = p2
                second_distance = point_distance
                second_match_desc = desc2
                continue

            if point_distance < nearest_distance:
                second_distance = nearest_distance
                second_match_point = nearest_match_point
                second_match_desc = nearest_match_desc
                nearest_distance = point_distance
                nearest_match_point = p2
                nearest_match_desc = desc2
            elif point_distance < second_distance:
                second_match_point = p2
                second_match_desc = desc2
            # Need to determine if we want a quicker computational approach

        # feature match
        distance_closest = distance(desc1, nearest_match_desc)
        distance_2ndclosest = distance(desc1, second_match_desc)
        match = distance_closest / distance_2ndclosest
        # if match < threshold then it is a match
        if match < threshold:
            match_list.append(match)
            distance_list.append(nearest_distance)
            point_list.append([p1, nearest_match_point, match, nearest_distance])
            cv2.line(test_img, p1[0].astype(int), ((nearest_match_point[0][0]+img1.shape[1]).astype(int),
                                                   nearest_match_point[0][1].astype(int)), (255, 0, 0))
            #draw.line((p1, (nearest_match[0]+img1.width, nearest_match[1])), fill=(255, 0, 0), width=7)
        #   Create a visual indication between the matched points for both images

    if len(point_list) > 0:
        output_path = Path(f'outputs/{runtime}/{int(threshold*100)}')
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(f'{output_path}/{image_a.name}_{image_b.name}.png', test_img):
            raise Exception("Image not saved.")

        raise Exception(type(point_list))
        match_sum = np.sum(point_list[:,2], dtype=object)
        raise Exception(str(point_list[:,2]) + '\n' + str(match_sum))
        nearest_sum = np.sum(point_list[:,3], dtype=object)
        return [True, image_a.name, image_b.name, match_sum, nearest_sum]
    else:
        return [False, image_a.name, image_b.name, np.inf, np.inf]
    #cv2.imwrite("lincoln-orb.jpg", img1)

def match_points(descs1, descs2):

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    return matcher.knnMatch(descs1, descs2, 2)

def orb_sift_match(image_a, image_b, threshold=0.99):

    global SAVE_IMG

    image_a, image_b = map(Path, [image_a, image_b])
    # Does some shit
    # Returns Boolean (True or False)
    # Possibly return distance to closest and second closest match too
    # E.g True/False
    # (True/False, distance, another_distance)

    #return '_3' in image_b or '_5' in image_b

    img1 = cv2.imread(str(image_a), cv2.IMREAD_GRAYSCALE)
    #img1 = cv2.imread(str(image_a))
    #print(img1.shape)
    img2 = cv2.imread(str(image_b), cv2.IMREAD_GRAYSCALE)
    #img2 = cv2.imread(str(image_b))
    #print(img2.shape)
    x_max = max(img1.shape[1], img2.shape[1])
    y_max = max(img1.shape[0], img2.shape[0])
    img1 = cv2.copyMakeBorder(img1, top=0, bottom=y_max-img1.shape[0], left=0, right=x_max-img1.shape[1], borderType=cv2.BORDER_CONSTANT)
    img2 = cv2.copyMakeBorder(img2, top=0, bottom=y_max-img2.shape[0], left=0, right=x_max-img2.shape[1], borderType=cv2.BORDER_CONSTANT)

    # you can increase nfeatures to adjust how many features to detect
    orb = cv2.ORB_create(nfeatures=1000)

    # detect features
    """    if image_a.name in descriptors and image_a.name in keypoints:
        (keypoints1, descriptors1) = keypoints[image_a.name], descriptors[image_a.name]
    else:
        (keypoints1, descriptors1) = orb.detectAndCompute(img1, None)
        keypoints[image_a.name] = keypoints1
        descriptors[image_a.name] = descriptors1
    if image_b.name in descriptors and image_b.name in keypoints:
        (keypoints2, descriptors2) = keypoints[image_b.name], descriptors[image_b.name]
    else:
        (keypoints2, descriptors2) = orb.detectAndCompute(img2, None)
        keypoints[image_b.name] = keypoints2
        descriptors[image_b.name] = descriptors2"""
    (keypoints1, descriptors1) = orb.detectAndCompute(img1, None)
    (keypoints2, descriptors2) = orb.detectAndCompute(img2, None)
    #keypoints1 = np.array([key_point.pt for key_point in keypoints1]).reshape(-1, 1, 2)
    #keypoints2 = np.array([key_point.pt for key_point in keypoints2]).reshape(-1, 1, 2)
    # Test image
    test_img = cv2.hconcat([img1, img2])

    desc_matches = match_points(descriptors1, descriptors2)
    output_path = Path(f'outputs/{runtime}/{int(threshold*100)}')
    output_img = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, desc_matches, None)
    if SAVE_IMG:
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(f'{output_path}/{image_a.name}_{image_b.name}.png', output_img):
            raise Exception(f'{output_path}/{image_a.name}_{image_b.name}.png')
    point_matches = []

    for closest, next_closest in desc_matches:

        ratio = closest.distance / next_closest.distance
        if ratio < threshold:
            point_matches.append((closest.queryIdx, closest.trainIdx, ratio, closest.distance))

    return point_matches

def check_pair_for_matches(pair):

    """
    pairings = case[1]
    threshold = case[0]
    matches = {pair: orb_sift_match(pair[0], pair[1], threshold=threshold) for pair in pairings}
    # Select only matched pairs
    #matched_pairs = [str(pair) for pair in matches if matches[pair]]
    matched_pairs = [f'{pair} | {len(matches[pair])}' for pair in matches]
    with open(f'{runtime}/matched_pairs_{threshold}.txt.', 'w+') as file:
        file.write('\n'.join(matched_pairs))
    """
    
    matches_ij = orb_sift_match(pair[0], pair[1])
    matches_ji = orb_sift_match(pair[1], pair[0])
    if not matches_ij or not matches_ji:
        sums = np.inf
    else:
        sums = (np.sum(np.array(matches_ij)[:,3]) + np.sum(np.array(matches_ji)[:,3])) / (len(matches_ij) + len(matches_ji))
    return (pair[0], pair[1], sums)

def cluster_images(images, k):

    # Get all possible pairs of images
    pairings = [pair for pair in itertools.product(images, images) if pair[0] != pair[1] and pair[0] < pair[1]]
    # Determine whether they have an ORB/SIFT Match
    #thresholds = [i / 100 for i in range(40,91,10)]
    #print(len(pairings), len(thresholds))
    matched_pairs = []
    with Pool(8) as p:
        #matched_pairs = tqdm(p.imap(generate_matched_pairs, cases), total=len(cases))
        for result in tqdm(p.imap_unordered(check_pair_for_matches, pairings), total=len(pairings)):
            matched_pairs.append(result)
    
    return matched_pairs

def main(images, output, k=10):

    #for i in range(0, 86, 5):
        #orb_sift_match("part1-images/eiffel_1.jpg", "part1-images/eiffel_1.jpg", threshold=i/100)
    clusters = cluster_images(images, k=k)
    with open('output.txt', 'w+') as file:
        file.write('\n'.join([str(value) for value in clusters]))


if __name__ == '__main__':
    # Step 1 Determine ORB Matching
    try:
        k = sys.argv[1]
        images = sys.argv[2:-1]
        output = sys.argv[-1]
    except:
        raise Exception(f'Usage: python3 part1.py <k>')

    freeze_support()
    main(images, output, k)
