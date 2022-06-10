import numpy as np
import matplotlib.pyplot as plt

def RGB(array):
    return array[:,:,0], array[:,:,1], array[:,:,2]

def segment_thresholding(image, roi):
    R,G,B = RGB(roi)
    r,g,b = RGB(image)
    meanR, meanG, meanB = np.mean(R), np.mean(G), np.mean(B)
    stdR, stdG, stdB = np.std(R), np.std(G), np.std(B)
    
    i = 1
    stdR, stdG, stdB = stdR * i, stdG * i, stdB * i
    
    pR = (r > (meanR - stdR)) & (r < (meanR + stdR))
    pG = (g > (meanG - stdG)) & (g < (meanG + stdG))
    pB = (b > (meanB - stdB)) & (b < (meanB + stdB))
    
    p = pR * pG * pB * 1
    p = p.astype(np.uint8) * 255
    return p

def PDF(x_img, x_crop):
    mu = np.mean(x_crop)
    sigma = np.std(x_crop)
    
    factor1 = (1 / (sigma * np.sqrt(2*np.pi)))
    factor2 = np.exp(-(x_img - mu)**2 / (2*(sigma**2)))
    
    return factor1*factor2

def segment_parametric(image,roi):
    R_img, G_img, B_img = RGB(image)
    R_img_crop1,G_img_crop1,B_img_crop1 = RGB(roi)

    P_R = PDF(R_img, R_img_crop1)
    P_G = PDF(G_img, G_img_crop1)
    P_B = PDF(B_img, B_img_crop1)
    
    joint_P_RGB = P_R * P_G *P_B
    joint_P_RGB = joint_P_RGB / joint_P_RGB.max()
    joint_P_RGB = (joint_P_RGB * 255).astype(np.uint8)
    return joint_P_RGB