import numpy as np
import cv2
def getHomography(kpsA, kpsB, matches,  reprojThresh):
    # convert the keypoints to np arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])

    if len(matches) > 4:
    # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])

        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5.0, reprojThresh, maxIters=2000, confidence=0.9)
        # H = cv2.getPerspectiveTransform(kpsA,kpsB)
        return (H, status)
    else:
        return None