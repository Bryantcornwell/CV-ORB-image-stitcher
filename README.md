# Assignment #2 Report
Group Members: Seth Mize, Lucas Franz, Bryant Cornwell

## Abstract

Due to the smartphone and Digital Age, image transformation operations have become more accessible. Image panoramics, image matching, etc. are a few applications of these operations. This report aims to match images using ORB feature point detection, perform image transformation operations to change image prospectives, and automatically stitch two similar images together utilizing RANSAC. The pairwise clustering accuracy for matching images yielded 82% utilizing OpenCV's brute force matching algorithm.

## Introduction

Utilizing OpenCV's ORB detection, we matched feature points between two images given a threshold to determine good matches. Paired images that matched well were grouped using an agglomerative clustering algorithm from the sklearn.cluster library and graded based on the pairwise clustering accuracy. 
In part 2 we are given two images of the same object with the objective of applying a transformation using a 3x3 matrix to one of the images to mirror the target image. To accomplish this we are given the necessary number of matching coordinate points in the image to solve for the transformation and then use inverse warping and bilinear interpolation to recreate the desired perspective. 
Finally, we combined the feature point matching along with RANSAC and image transformations to blend two images together within the first image's coordinate system. 

## Methods
### part1.py
Run the code from the terminal using the following format on the linux server and ensure to type the required parameter for the desired arguments below:

    ./a2 part1 <k> images/*.png outputfile_txt

The first task of part1.py was to load two images using OpenCV library and create a border around each image based on the largest image width and height. The border was used for image concatenation when testing both images side by side. 
An orb object is created to detect 1000 features within each image and extract points & descriptors for each feature. 
To check for a match, you iterate through all pairing of orbs between image A and image B and for each pairing we calculate the hamming distance. 
Next we go through each orb and determine the nearest and second nearest descriptors in order to determine if the match is good or not.
The final matching algorithm we used to perform this matching is a brute force matcher from opencv (cv2.BFMatcher()).
To check for a good match, the ratio of closest and next_closest hamming distances compared to a threshold of 0.75 for each match. If the ratio is lower than the threshold, it is considered a good match and the corresponding points are passed to a list.

The assignment hinted the use of agglomerative clustering since we had pairwise distances between objects and not a feature for each image.
Given the input number of clusters, k, and the input images, we fit the orb pair distances utilizing sklearn's agglomerative clustering algorithm to generate clusters. 
Each cluster is added to a dictionary and used to determine the pairwise clustering accuracy.
The accuracy is computed by adding the number of true positive and true negatives together and dividing by the total number of possible pairs of images. 
A pair is a true positive is when a pair has the same image name and is in the same cluster.
A pair is a true negative is when a pair does not have the same image name and is not in the same cluster.
This part concludes by printing the pairwise clustering accuracy and writing the clusters separated by a newline to an output.txt file.


### part2.py
Run the code from the terminal using the following format on the linux server and ensure to type the required parameter for the desired arguments below:

    ./a2 part2 n img_1.png img_2.png img_output.png img1_x1,img1_y1 img2_x1,img2_x1 ... img1_xn,img1_yn img2_xn,img2_yn

This program relies upon the passed argument for the type of transformation that will be solved for. There are four types of transformations that can be passed using the values for 'N' of 1, 2, 3, 4.

- 1 = Translation
- 2 = Euclidean
- 3 = Affine
- 4 = Projection

Given the passed value of 'N', an equal number of matching coordinate pairs must be pass ((beginning x, beginning y), (target x, target y)). The program first ensures coordinate matches are included and then validates that there is an appropriate number of pairs to solve for the desired transformation.

Four separate functions are used to solve for the 4 different transformation. The following diagrams from Module 7 Image Transformation powerpoint slides [3] and 2D Projective Geometry theory resource [4] were used to create the corresponding linear system of equations to solve for the underlying variables representing the degrees of freedom for each transformation:

| 2D Projective Geometry theory resource [4] | 
| :-----------------------------------------------------------------------------------: | 
| <img src="documentation/images/Transformation_Matrix_Diagram.jpg" alt="image_name" width="400"/> |  

The solved transformation matrix is the inverse transformation which leads to the next step of the code of applying inverse warping to the image that we are trying to transform into the perspective of the target image. In inverse warping we loop through the pixels of the transformed image, apply the inverse transform matrix, and lookup the corresponding coordinate from the original image. Applying forward warping has the potential to give a fractional destination pixel, resulting in holes in the image. Using inversing warping we still get a fractional original coordinate, but we can then apply bilinear interpolation to get a weighted contribution from the surrounding original pixels to create a new proportional pixel.

### part3.py
Run the code from the terminal using the following format on the linux server and ensure to type the required parameter for the desired arguments below:

    ./a2 part3 image_1.jpg image_2.jpg output.jpg

This program utilizes functions from part1.py and part2.py, specifically;

- part1.py: orb_sift_match, pad_image
- part2.py: get_projection_matrix, test_transition_matrix, apply_transformation

Two images are passed to the program and read into numpy arrays. The orb_sift_match function is then applied to extract feature point matches between the two images. 

Once the feature point matches are gathered we apply random sample consensus (RANSAC) to determine the appropriate transformation matrix. To do this, we take the set of feature point matches and sample a set number of matches from them for a passed number of experiments. For each experiment, the feature points are shuffled to create a randomized sample each time. The sampled points are then used to solve for the projection matrix given the points. The remaining feature point matches that were not used to solve for the projection matrix are then used to test the found projection matrix to see how many of the feature point matches are in agreement with the found matrix. 

The sample with the lowest test error is preserved and passed to the apply_transformation fuction. This function applied the projection appropriately to one of the images to bring both images into the same perspective and prepare them to be merged.

(Find centroid and merge images)


## Results

### part1.py

The pairwise clustering accuracy from the part1 feature matching algorithm give 82%.

### part2.py

To judge our results on part 2 of the assignment, we were given 2 test cases.

The first test case was to solve for inverse warping and bilinear interpolation using an image of the lincoln memorial from one perspective and a given transformation matrix to result in a new perspective. Displayed below are the results of implementing this process.

| Input | Projection |
| :-----------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |  
| <img src="documentation/images/lincoln.jpg" alt="image_name" width="200"/> | <img src="documentation/images/lincoln_output.jpg" alt="image_name" width="200"> | 

The second test case involved testing our solved transformation matrix for projection. For this we were given 2 images, the target perspective and the input perspective, as well as a list of four matching coordinates. Displayed below are the results generated by using the four points to solve for the transformation matrix and applying the projection through inverse warping and bilinear interpolation.

| Target | Input | Projection |
| :-----------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |  :---------------------------------------------------------------------------------------------------: | 
| <img src="documentation/images/book1.jpg" alt="image_name" width="200"/> | <img src="documentation/images/book2.jpg" alt="image_name" width="200"> | <img src="documentation/images/book3.jpg" alt="image_name" width="200"> |


## Discussion
### part1.py
There were some difficulties with that were encountered in the early stages of the orb matching algorithm.
The first version of the algorithm was written based on the information gather from the image and feature matching video in Module 6 Feature Points [1]. 
The initial debugging consistent of changing the keypoint data type in order to carry out mathematical operations. 
The first visualization of the feature point matching algorithm in the figure below gave an idea of apparent semantic issues. 
Utilizing a high threshold, many of the orbs from the first image were mapped to similar location(s). 
The root causes were that the second nearest matches were not updated correctly, and the keypoints were used as the descriptors in the distance calculations.

![Phase1_orbmatch.png](documentation/images/example_match_100_20220326133138.png)

Using the final version of the ORB matching algorithm, we received the following result comparing bigben_2.jpg with bigben_10.jpg.

![Phase1_orbmatch.png](documentation/images/bigben_2.jpg_bigben_10.jpg.png)

The runtime of part1.py using the orb matching algorithm took around 3 hours to run with multiprocessing enabled. 
While looking through the Q&A board, the class was given permission to use the cv2.BFMatcher() function from the openCV library [2]. 
This decreased the computational runtime to around 3 minutes. 
We used sklearn's agglomerative clustering algorithm to generate clusters as a suggestion from the professor. Agglomerative clustering is a type of hierarchical clustering where in our case each pair are placed into ten different groups based on the distances of the pairs [6].


Overall, part1.py could be further improved by hyperparameter tuning the clustering algorithm and feature matching threshold. Future improvements to the original feature matching algorithm could provide promising results, but could result in a long development due to the current computational runtime.

### part2.py

To start the problem we were given an image and corresponding transformation matrix and what our result should look like. This was extremely helpful in initially configuring the inverse warping and bilinear interpolation portions of the code. Once that was accomplished we were able to focus on writing the code for solving for the transformation matrix given a set of coordinate pairs.

In order to construct this code we needed to design a simple image and apply different types of transformations to it to be able to test the different circustances our code may be tested with. Our simple image was a square consisting of a 1 pixel border on a black background. We then apply all four types of transformations to it. Once we had produced the image, we could use the transformation matrix to generate sets of coordinate pairs. Depending on the transformation applied, we then began creating transformation solve functions and limited their input to only the minimum number of coordinate pairs needed to solve for that type of transformation. We were then able to test the transformation matrix by applying the inversing warping to see if we could recreate the original square.

Table of Image transformations:

| Transformation | Original | Transformed | Reconstructed |
| :-----------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |  :---------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |
| Translation | <img src="documentation/images/Simple.jpg" alt="image_name" width="200"/> | <img src="documentation/images/Simple_Translation.jpg" alt="image_name" width="200"> | <img src="documentation/images/Simple_Translation_Inverse.jpg" alt="image_name" width="200"> |
| Euclidean | <img src="documentation/images/Simple.jpg" alt="image_name" width="200"/> | <img src="documentation/images/Simple_Euclidean.jpg" alt="image_name" width="200"> | <img src="documentation/images/Simple_Euclidean_Inverse.jpg" alt="image_name" width="200"> |
| Affine | <img src="documentation/images/Simple.jpg" alt="image_name" width="200"/> | <img src="documentation/images/Simple_Affine.jpg" alt="image_name" width="200"> | <img src="documentation/images/Simple_Affine_Inverse.jpg" alt="image_name" width="200"> |
| Projection | <img src="documentation/images/Simple.jpg" alt="image_name" width="200"/> | <img src="documentation/images/Simple_Project.jpg" alt="image_name" width="200"> | <img src="documentation/images/Simple_Project_Inverse.jpg" alt="image_name" width="200"> |


We did have to spend additional time working through the linear algebra in expanding the matrix multiplication given the diagrams referenced in part2.py methods, so that we could appropriately configure the equation matrix and solution matrix to be utilized by the numpy.linalg.solve() function. A creative solution was derived for projective transformation in which additional variables are solved for, but not used when constructing the transformation matrix.

Additional work we could have done to this code would have been to create a dynamic function to solve for the transformation matrix instead of four separate functions. We could also look into the running time of applying the transformation as it is the longest running time of our code. Future work on the overall problem would involve looking into the skewing we are seeing after applying the inverse warping using a projective transformation matrix. 

### part3.py

Attempt 1: 

Mapping a centroid to the coordination system in Image A from Image B.

| Image A Centroid | Image B Centroid  | Output |
| :-----------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |  :---------------------------------------------------------------------------------------------------: | 
| <img src="documentation/part3/attempt1/image_a_centroid.jpg" alt="image_name" width="200"/> | <img src="documentation/part3/attempt1/image_b_centroid.jpg" alt="image_name" width="200"> | <img src="documentation/part3/attempt1/book3.jpg" alt="image_name" width="200"> |

Attempt 2:

Correctly mapping an Image B centroid to Image A coordinate system.

| Image A Centroid | Image B Centroid  | Output |
| :-----------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |  :---------------------------------------------------------------------------------------------------: | 
| <img src="documentation/part3/attempt2/image_a_centroid.jpg" alt="image_name" width="200"/> | <img src="documentation/part3/attempt2/image_b_centroid.jpg" alt="image_name" width="200"> | <img src="documentation/part3/attempt2/book3.jpg" alt="image_name" width="200"> |

Attempt 3:

Applying the algorithm from attempt 2 to our images. Notice nearly half of the transformed image is cut off.

| Image A Centroid | Image B Centroid  | Output |
| :-----------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |  :---------------------------------------------------------------------------------------------------: | 
| <img src="documentation/part3/attempt3/image_a_centroid.jpg" alt="image_name" width="200"/> | <img src="documentation/part3/attempt3/image_b_centroid.jpg" alt="image_name" width="200"> | <img src="documentation/part3/attempt3/image_b_t_centroid.jpg" alt="image_name" width="200"> |

Attempt 4:

Trying to fit the transformation inside of the image by extending the image boundary.

| Image A Centroid | Image B Centroid  | Output |
| :-----------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |  :---------------------------------------------------------------------------------------------------: | 
| <img src="documentation/part3/attempt4/image_a_centroid.jpg" alt="image_name" width="200"/> | <img src="documentation/part3/attempt4/image_b_centroid.jpg" alt="image_name" width="200"> | <img src="documentation/part3/attempt4/image_b_t_centroid.jpg" alt="image_name" width="200"> |

Attempt 5:

Correctly fit the transformed image into the image boundary.

| Image A Centroid | Image B Centroid  | Output |
| :-----------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |  :---------------------------------------------------------------------------------------------------: | 
| <img src="documentation/part3/attempt5/image_a_centroid.jpg" alt="image_name"/> | <img src="documentation/part3/attempt5/image_b_centroid.jpg" alt="image_name"> | <img src="documentation/part3/attempt5/image_b_t_centroid.jpg" alt="image_name"> |


## Conclusions

## Acknowledges
### Bryant Cornwell 
Co-wrote and tested part1.py with Seth. Contributed to discussions on implementing part2.py and part3.py. 
For the report, wrote the introduction, abstract, part1.py methods, part1.py results, part1.py discussion, and general layout of report.
### Seth Mize
### Lucas Franz
Wrote part2.py inverse warping and bilinear interpolation code. Created "Simple" test cases for work solving different transformations. Implemented transformation solving code provided by Seth. In the report, wrote part2.py methods, results, discussion, and contributed to part3.py methods. 

## References
[1] Module 6.7 video: https://iu.instructure.com/courses/2032639/pages/6-dot-7-image-and-feature-matching?module_item_id=25895156

[2] Theory on opencv Matcher: https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

[3] Module 7 slides: https://iu.instructure.com/courses/2032639/files/133487313/download?wrap=1

[4] 2D Projective Geometry reference: https://fzheng.me/2016/01/14/proj-transformation/

[5] RANSAC reference: https://en.wikipedia.org/wiki/Random_sample_consensus

[6] Hierarchical agglomerative clustering: https://en.wikipedia.org/wiki/Hierarchical_clustering

#To complete:
- (optional) Seth review Part1 Methods to add any recent changes made.
- Part 3 Methods and Discussion Sub-Sections
- Results Section (Add clusters for part1, and image results for part 3)
- Conclusion Section
- Finalize Acknowledgements Section
- Add references to documentation to proper sections of the report