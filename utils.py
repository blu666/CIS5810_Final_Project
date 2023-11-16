import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

def findHomography(img1, img2, min_match=10, showMatches=False):
    """
    Find homography between image 1 and image 2

    :return H: homography matrix 3x3 
    """
    img_gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img_gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img_gray1,None)
    kp2, des2 = sift.detectAndCompute(img_gray2,None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches according to ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > min_match:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        # h,w = img_gray1.shape
        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # dst = cv2.perspectiveTransform(pts,M)
        # img_gray2 = cv2.polylines(img_gray2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), min_match) )
        matchesMask = None

    if showMatches:
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)
        img_match = cv2.drawMatches(img_gray1,kp1,img_gray2,kp2,good,None,**draw_params)
        img_match = cv2.cvtColor(img_match, cv2.COLOR_RGB2BGR)
        cv2.imwrite('result/match.jpg', img_match)
        # img_match = Image.fromarray((np.clip(img_match, 0, 1) * 255).astype(np.uint8))
        # img_match.save('result/match.jpg')

    return M

def findBoundingBox(img):
    """
    Find bounding box of single person in image

    :return bounding_boxes: bounding boxes of persons in image
    """
    # initialize the HOG descriptor
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # detect humans in input image
    (humans, _) = hog.detectMultiScale(img, winStride=(10, 10),
                                    padding=(32, 32), scale=1.05)

    # Initialize an empty list to store bounding box coordinates
    bounding_boxes = []

    # loop over all detected humans
    for (x, y, w, h) in humans:
        pad_w, pad_h = int(0.15 * w), int(0.01 * h)
        # Store the bounding box coordinates
        bbox = [x + pad_w, y + pad_h, x + w - pad_w, y + h - pad_h]
        bounding_boxes.append(bbox)

    return np.array(bounding_boxes[0])


def findMask(img):
    """
    Find mask of single person in image

    :return mask: mask of image
    """
    bboxes = findBoundingBox(img)
    sam = sam_model_registry["vit_h"](checkpoint="sam_ckpt/sam_vit_h_4b8939.pth")

    predictor = SamPredictor(sam)
    predictor.set_image(img)
    masks, _, _ = predictor.predict(
                                    point_coords=None,
                                    point_labels=None,
                                    box=bboxes[None, :],
                                    multimask_output=False,
                                )
    return masks, bboxes


def poissonBlending(img1, img2, mask):
    """
    Poisson blending of masked region in image 1 and image 2
    
    :return res: blended image
    """

def plt_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def plt_bbox(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

if __name__ == "__main__":
    img_path = 'Images'
    img_name = 'group'
    img_suffix = '.jpg'
    img1 = cv2.imread(img_path + '/' + img_name + '_1' + img_suffix)
    img2 = cv2.imread(img_path + '/' + img_name + '_2' + img_suffix)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    H = findHomography(img1, img2, showMatches=True)
    # print(H)

    # _, axs = plt.subplots(2, 2)
    # axs[0, 0].imshow(img1)
    # axs[0, 1].imshow(img2)
    # axs[1, 0].imshow(cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0])))
    # # axs[1, 1].imshow()
    # plt.show()
    warped_img1 = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))
    # cv2.imwrite('result/warped_img1.jpg', cv2.cvtColor(warped_img1, cv2.COLOR_RGB2BGR))
    # print(warped_img1.shape)
    # bbox = findBoundingBox(warped_img1)
    # print(bbox)
    # masks, bbox = findMask(warped_img1)
    # np.save('tmp/mask.npy', masks[0])
    # np.save('tmp/bbox.npy', bbox)

    # print(bbox)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(warped_img1)
    # plt_mask(masks[0], plt.gca())
    # plt_bbox(bbox, plt.gca())
    # plt.axis('off')
    # plt.show()

    masks = np.load('tmp/mask.npy')
    # print(np.sum(masks))
    # bbox = np.load('tmp/bbox.npy')
    new_img = img2.copy()
    new_img[:,:,0][np.where(masks)] = (warped_img1[:,:,0][np.where(masks)]).flatten()
    new_img[:,:,1][np.where(masks)] = (warped_img1[:,:,1][np.where(masks)]).flatten()
    new_img[:,:,2][np.where(masks)] = (warped_img1[:,:,2][np.where(masks)]).flatten()
    # new_image[:,:,1] = masks * warped_img1[:,:,1]
    # new_img[:,:,0] = np.where(masks[0], img2[:,:,0], warped_img1[:,:,0])
    # new_img[:,:,1] = np.where(masks[0], img2[:,:,1], warped_img1[:,:,1])
    # new_img[:,:,2] = np.where(masks[0], img2[:,:,2], warped_img1[:,:,2])
    new_img.reshape(600, 800, 3)
    plt.imshow(new_img)
    cv2.imwrite('result/combined_img.jpg', cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
    plt.show()