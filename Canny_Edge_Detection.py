from scipy.ndimage.filters import convolve as convolv
import cv2
import numpy as np

def get_gaussian_filter(size, sigma):
    size = int(size/2)
    x, y = np.mgrid[-size:size+1,-size:size+1]
    normal = 1/(2.0*np.pi*(sigma**2))
    g =  np.exp(-((x**2+y**2)/(2.0*sigma**2)))*normal
    return g

def find_gradient_using_sobel_filter(img):
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True)
    return (mag,ang)

def reduce_noise(img,filter_size,sigma):
    return convolv(img,get_gaussian_filter(filter_size, sigma))

def non_maximal_supression(mag, ang):
    height, width = mag.shape
    for i in range(1,height-1):
        for j in range(1,width-1):
            neighbour1 = 255
            neighbour2 = 255

            ang[i][j] = abs(ang[i][j]-180) if abs(ang[i][j])>180 else abs(ang[i][j])
            # - direction
            if (0 <= ang[i,j] < 22.5) or ((22.5+135) <= ang[i,j] <= 180):
                neighbour1 = mag[i][j+1]
                neighbour2 = mag[i][j-1]
            # / direction
            elif (22.5 <= ang[i,j] < (22.5+45)):
                    neighbour1 = mag[i-1][j-1]
                    neighbour2 = mag[i+1][j+1]
                # | direction
            elif ((22.5+45) <= ang[i,j] < (22.5+90)):
                neighbour1 = mag[i+1][j]
                neighbour2 = mag[i-1][j]
            # \ direction
            elif ((22.5+90) <= ang[i,j] < (22.5+135)):
                neighbour1 = mag[i+1][j-1]
                neighbour2 = mag[i-1][j+1]
            else:
                pass

            if ((mag[i][j] < neighbour1) or (mag[i][j] < neighbour2)):
                mag[i][j]=0

    return mag

def double_thresholding(mag,low_T,high_T,weak_pixel_val,strong_pixel_val):
    high_T = mag.max()*high_T
    low_T = high_T*low_T

    strong_x, strong_y = np.where(mag >= high_T)
    weak_x, weak_y = np.where((mag < high_T) & (mag > low_T))

    temp = np.zeros_like(mag)
    temp[strong_x, strong_y] = strong_pixel_val
    temp[weak_x, weak_y] = weak_pixel_val

    return temp

def hysteresis(img,weak_pixel_val,strong_pixel_val):
    height, width = img.shape
    for i in range(1, height-1):
        for j in range(1, width-1):
            if (img[i,j] == weak_pixel_val):
                if (
                       (img[i+1, j-1] == strong_pixel_val)
                    or (img[i+1, j] == strong_pixel_val)
                    or (img[i+1, j+1] == strong_pixel_val)
                    or (img[i, j-1] == strong_pixel_val)
                    or (img[i, j+1] == strong_pixel_val)
                    or (img[i-1, j-1] == strong_pixel_val)
                    or (img[i-1, j] == strong_pixel_val)
                    or (img[i-1, j+1] == strong_pixel_val)
                ):
                    img[i, j] = strong_pixel_val
                else:
                    img[i, j] = 0
    return img

def canny_edge_detection(img,filter_size,sigma,weak_pixel_val,strong_pixel_val,low_T,high_T):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = reduce_noise(img,filter_size,sigma)
    gradient_magnitude, gradient_angle = find_gradient_using_sobel_filter(img)
    gradient_magnitude = non_maximal_supression(gradient_magnitude,gradient_angle)
    img = double_thresholding(gradient_magnitude,low_T,high_T,weak_pixel_val,strong_pixel_val)
    img = hysteresis(img,weak_pixel_val,strong_pixel_val)
    return img

# img = cv2.imread('lane.jpeg',0)
# output=canny_edge_detection(img,5,1.4,100,255,0.1,0.5)
# cv2.imshow("==== output ====",output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
