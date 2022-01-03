import numpy as np
import cv2
import os
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
from homo import *
from stitch import stitch

def  stitch_images(args, image1, feat1, image2, feat2, count, len_input_image_list):
    matches = match_descriptors(feat1['descriptors'], feat2['descriptors'], cross_check=True)
    print('Number of raw matches: %d.' % matches.shape[0])

    keypoints_left = feat1['keypoints'][matches[:, 0], : 2]
    keypoints_right = feat2['keypoints'][matches[:, 1], : 2]
    
    np.random.seed(0)
    model, inliers = ransac(
        (keypoints_left, keypoints_right),
        ProjectiveTransform, min_samples=4,
        residual_threshold=4, max_trials=10000
    )
    n_inliers = np.sum(inliers)
    print("number os inliers: %d." % n_inliers)
    inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_left[inliers]]
    inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_right[inliers]]

    placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
    image3 = cv2.drawMatches(image1, inlier_keypoints_left, image2, inlier_keypoints_right, placeholder_matches, None)
    cv2.imwrite(os.path.join('matches','Match')+str(count//2)+'.jpg', image3)
    
    #finding the homography
    M = getHomography(inlier_keypoints_left, inlier_keypoints_right, placeholder_matches, reprojThresh=4)
    if M is None:
        print("Error")
    (H, status) = M

    #Stitch the given two images together
    stitch(image1, image2, H, count, len_input_image_list)