import cv2


def findHomography(img1, img2):
    """
    Find homography between image 1 and image 2

    :return H: homography matrix 3x3 
    """
    pass


def findMask(img):
    """
    Find mask of single person in image

    :return mask: mask of image
    """
    pass


def poissonBlending(img1, img2, mask):
    """
    Poisson blending of masked region in image 1 and image 2
    
    :return res: blended image
    """