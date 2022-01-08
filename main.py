import cv2
import numpy as np
import random


def solution(left_img, right_img):
    kp1, des1, kp2, des2 = extract_keypoint(left_img, right_img)

    good_matches = match_keypoint(kp1, kp2, des1, des2)

    final_H = ransac(good_matches)

    result_img = wrap_and_stitch(left_img, right_img, final_H)

    return result_img
    

def calculate_minimum_distance(i, d1, des2):
    k = 2
    dist = []
    j = 0
    for d2 in des2:
        dist.append([i, j, np.linalg.norm(d1 - d2)])
        j = j + 1
    dist.sort(key=lambda x: x[2])
    return dist[0:k]

def extract_keypoint(left_img, right_img):
    gray1 = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    surf = cv2.xfeatures2d.SURF_create()
    sift = cv2.xfeatures2d.SIFT_create()

    kp1 = sift.detect(gray1, None)
    kp1, des1 = surf.compute(gray1, kp1)

    kp2 = sift.detect(gray2, None)
    kp2, des2 = surf.compute(gray2, kp2)
    return kp1, des1, kp2, des2

def match_keypoint(kp1, kp2, des1, des2):
    #knn
    i = 0
    total_matches = []
    for d1 in des1:
        total_matches.append(calculate_minimum_distance(i, d1, des2))
        i = i + 1

    # ratio test
    good_matches = []
    for m, n in total_matches:
        if m[2] < 0.75*n[2]:
            left_pt = kp1[m[0]].pt
            right_pt = kp2[m[1]].pt
            good_matches.append(
                [left_pt[0], left_pt[1], right_pt[0], right_pt[1]])
    return good_matches

def compute_homography(points):
    A = []
    for pt in points:
      x, y, x_dash, y_dash = pt[0], pt[1], pt[2], pt[3]
      A.append([x, y, 1, 0, 0, 0, -1 * x_dash * x, -1 * x_dash * y, -1 * x_dash])
      A.append([0, 0, 0, x, y, 1, -1 * y_dash * x, -1 * y_dash * y, -1 * y_dash])

    A = np.array(A)
    u, s, vh = np.linalg.svd(A)
    H = vh[-1, :]
    H = H.reshape(3, 3)
    H = H / H[2, 2]
    return H

def compute_inliers(pts, H):
    inliers = []
    t = 5
    for pt in pts:
        p = np.array([pt[0], pt[1], 1]).reshape(3, 1)
        Hp = np.dot(H, p)
        Hp = Hp / Hp[2]
        p_dash = np.array([pt[2], pt[3], 1]).reshape(3, 1)
        dist = np.linalg.norm(p_dash - Hp)

        if dist < t:
            inliers.append(pt)
    return inliers

def ransac(pts):
    best_inliers = []
    final_H = []
    for i in range(5000):
        random_pts = random.choices(pts, k=4)
        H = compute_homography(random_pts)
        inliers = compute_inliers(pts, H)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            final_H = H
    return final_H

def wrap_and_stitch(left_img, right_img, final_H):
    r1, c1 = right_img.shape[:2]
    r2, c2 = left_img.shape[:2]
    list1 = np.float32([[0,0], [0, r1],[c1, r1], [c1, 0]]).reshape(-1, 1, 2)
    list2 = np.float32([[0,0], [0, r2], [c2, r2], [c2, 0]]).reshape(-1, 1, 2)
    list3 = cv2.perspectiveTransform(list2, final_H)
    final_list = np.concatenate((list1, list3), axis=0)
    [x_min, y_min] = final_list.min(axis=0).flatten().astype(np.int32)
    [x_max, y_max] = final_list.max(axis=0).flatten().astype(np.int32)
    dist = [-x_min, -y_min]
    H = np.array([[1, 0, dist[0]], [0, 1, dist[1]], [0, 0, 1]])
    result_img = cv2.warpPerspective(left_img, H.dot(final_H), (x_max - x_min, y_max - y_min))
    result_img[dist[1]:r1 + dist[1], dist[0]:c1 + dist[0]] = right_img
    return result_img

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('result.jpg', result_img)


