import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob

def load_images(filename1, filename2, rescale):
    img1 = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
    
    img1_res = cv2.resize(img1, dsize = (img1.shape[1]//rescale, img1.shape[0]//rescale), interpolation=cv2.INTER_CUBIC)
    img2_res = cv2.resize(img2, dsize = (img2.shape[1]//rescale, img2.shape[0]//rescale), interpolation=cv2.INTER_CUBIC)
    
    return img1_res, img2_res

def find_matching_points(img1,img2, detector):
    if detector == 'SIFT':
        feature_detector = cv2.SIFT_create()
    if detector == 'ORB':
        feature_detector = cv2.ORB_create()
    key1,des1 = feature_detector.detectAndCompute(img1, None)
    key2,des2 = feature_detector.detectAndCompute(img2, None)
    
    #Use Flann-Based Matcher
    if detector =='SIFT':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
    if detector =='ORB':
        FLANN_INDEX_LSH = 6
        index_params =dict(algorithm=FLANN_INDEX_LSH,table_number=6,key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1, des2, k = 2)
    
    return key1,key2,des1,des2,matches


def draw_matches(img1,img2,key1,key2,good_matches,matches,limit):
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=good_matches[:limit],
                   flags=cv2.DrawMatchesFlags_DEFAULT)

    keypoint_matches = cv2.drawMatchesKnn(img1, key1, img2, key2, matches[:limit], None, **draw_params)
    plt.figure(figsize = (20,10))
    plt.imshow(keypoint_matches)
    plt.axis('off')
    plt.show()
    
def draw_keypoints(img1, img2, key1, key2):
    output_image1 = cv2.drawKeypoints(img1, key1, img1, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    output_image2 = cv2.drawKeypoints(img2, key2, img2, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    plt.figure(figsize = (10,20))
    plt.subplot(121)
    plt.imshow(output_image1)
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(output_image2)
    plt.axis('off')
    plt.show()
    
def estimate_fundamental_matrix(key1, key2, matches, constant):
    good_matches = [[0, 0] for i in range(len(matches))]
    pts1, pts2 = [], []
    for i, (m,n) in enumerate(matches):
        if m.distance < constant * n.distance:
            good_matches[i] = [1, 0]
            pts1.append(key1[m.queryIdx].pt)
            pts2.append(key2[m.trainIdx].pt)
    
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, inliers = cv2.findFundamentalMat(pts1, pts2, method = cv2.FM_RANSAC)
    
    pts1 = pts1[inliers.ravel() == 1]
    pts2 = pts2[inliers.ravel() == 1]
    return pts1, pts2, good_matches, F, inliers

def drawlines(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1src.shape
    img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
    img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        color2 = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv2.circle(img2color, tuple(pt2), 5, color2, -1)
    return img1color, img2color

def find_epilines(img_l,img_r,pts1,pts2,F):
    linesLeft = cv2.computeCorrespondEpilines(np.array(pts2).reshape(-1, 1, 2), 2, F)
    linesLeft = np.array(linesLeft).reshape(-1, 3)
    img5, img6 = drawlines(img_l, img_r, linesLeft, pts1, pts2)

    linesRight = cv2.computeCorrespondEpilines(np.array(pts1).reshape(-1, 1, 2), 1, F)
    linesRight = np.array(linesRight).reshape(-1, 3)

    img3, img4 = drawlines(img_r, img_l, linesRight, pts2, pts1)
    
    return img5, img3

def rectify_images(img1,img2,pts1,pts2,F, thresh):
    # Stereo rectification (uncalibrated variant)
    # Adapted from: https://stackoverflow.com/a/62607343
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1), threshold = thresh)
    
    img1_rec = cv2.warpPerspective(img1, H1, (w1, h1))
    img2_rec = cv2.warpPerspective(img2, H2, (w2, h2))
    
    return img1_rec, img2_rec

def rectify_images2(img1,img2,pts1,pts2,F):
    h,status = cv2.findHomography(pts1, pts2)
    im_rect = cv2.warpPerspective(img1, h,(img2.shape[1], img2.shape[0]))
    return im_rect

def get_disparity_map(img1, img2, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities = numDisparities, blockSize=blockSize)
    disparity = stereo.compute(img1, img2)
    
    return disparity

