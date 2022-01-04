import numpy as np
import cv2
import os
import imutils


def crop(panorama, h_dst, conners):
    """crop panorama based on destination.
    @param panorama is the panorama
    @param h_dst is the hight of destination image
    @param conner is the tuple which containing 4 conners of warped image and
    4 conners of destination image"""
    # find max min of x,y coordinate
    [xmin, ymin] = np.int32(conners.min(axis=0).ravel() - 0.5)
    t = [-xmin, -ymin]
    conners = conners.astype(int)

    # conners[0][0][0] is the X coordinate of top-left point of warped image
    # If it has value<0, warp image is merged to the left side of destination image
    # otherwise is merged to the right side of destination image
    if conners[0][0][0] < 0:
        n = abs(-conners[1][0][0] + conners[0][0][0])
        panorama = panorama[t[1] : h_dst + t[1], n:, :]
    else:
        if conners[2][0][0] < conners[3][0][0]:
            panorama = panorama[t[1] : h_dst + t[1], 0 : conners[2][0][0], :]
        else:
            panorama = panorama[t[1] : h_dst + t[1], 0 : conners[3][0][0], :]
    return panorama

def stitch(image1, image2, H, count, len_input_image_list):    
    if count<len_input_image_list:
        rows1, cols1 = image2.shape[:2]
        rows2, cols2 = image1.shape[:2]
    else:
        rows1, cols1 = image1.shape[:2]
        rows2, cols2 = image2.shape[:2]


    list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)

    # When we have established a homography we need to warp perspective
    # Change field of view
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

    list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel())
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel())
    
    translation_dist = [-x_min,-y_min]
    
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    if count<len_input_image_list:
        output_img = cv2.warpPerspective(image1, H_translation.dot(H), (x_max-x_min, y_max-y_min))
        print(output_img.shape)
        output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = image2
    else:
        output_img = cv2.warpPerspective(image2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
        print(output_img.shape)
        output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = image1
    
    gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

	# Finds contours from the binary image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

	# get the maximum contour area
    c = max(cnts, key=cv2.contourArea)

	# get a bbox from the contour area
    (x, y, w, h) = cv2.boundingRect(c)

	# crop the image to the bbox coordinates
    output_img = output_img[y:y + h, x:x + w]
    # output_img = crop(output_img, output_img.shape[1], list_of_points)
    cv2.imwrite(os.path.join('stitched_images','SI') + str(count+1) + '.jpg', output_img)