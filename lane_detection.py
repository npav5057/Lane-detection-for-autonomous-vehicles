import cv2
import numpy as np
import argparse
import sys
from Canny_Edge_Detection import canny_edge_detection as my_canny

def canny_edge_detector(img):
    canny_img = my_canny(img,5,1.4,100,255,0.1,0.5)
    return canny_img

def get_mask(h,w,th,tv,tm):
    polygons = np.array([[(round((1-th)*w), round(h)), (round(w/2), round(tm*h)), (round(th*w), round(h))]])
    mask = np.zeros((h, w), np.float32)
    cv2.fillPoly(mask, polygons, 255)
    for i in range(round(h*tv),h):
        for j in range(w):
            mask[i][j]=255
    return mask

def get_coordinate_from_slope_and_y_intercept (h,slope,intercept):
    y1 = h
    y2 = 0
    x1 = round((y1-intercept)/slope)
    x2 = round((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

def extrapolate_line(h,w,lines):
    left_line = []
    right_line = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        params = np.polyfit((x1, x2), (y1, y2), 1)
        M = params[0]
        C = params[1]
        if M < 0:
            left_line.append((M, C))
        else:
            right_line.append((M, C))

    # if (len(left_line)==0 or len(right_line)==0):
    #     print("ERROR: Both lint are not detected instead only one line is getting detected.")
    #     exit(0)

    left_avg_line = np.average(left_line, axis = 0)
    right_avg_line = np.average(right_line, axis = 0)

    final_left_line = get_coordinate_from_slope_and_y_intercept(h,left_avg_line[0],left_avg_line[1])
    final_right_line = get_coordinate_from_slope_and_y_intercept(h,right_avg_line[0],right_avg_line[1])

    if(left_avg_line[0]!=right_avg_line[0]):
        I_x=round((left_avg_line[1]-right_avg_line[1])/(right_avg_line[0]-left_avg_line[0]))
        I_y=round((I_x*left_avg_line[0])+left_avg_line[1])
        final_left_line[0+2]=min(h,I_x)
        final_left_line[1+2]=min(h,I_y)
        final_right_line[0+2]=min(h,I_x)
        final_right_line[1+2]=min(h,I_y)

    return np.array([final_left_line, final_right_line])

def draw_lane_on_image(img, lines, thickness_of_lane):
    line_img = np.zeros_like(img)
    color_of_lane=[0, 0, 255]
    if lines is not None:
        for x1, y1, x2, y2 in lines:
                    cv2.line(line_img, (x1, y1), (x2, y2), color_of_lane, thickness_of_lane)

    resultant = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return resultant

def get_lines(img):
    lines = cv2.HoughLinesP(
                                img,
                                rho=2,
                                theta=np.pi/180,
                                threshold=100,
                                minLineLength=40,
                                maxLineGap=25
                            )
    return lines


print('\n********************* Lane Detection ******************\n\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lane detection')
    parser.add_argument('--input_image', type=str,  help='path of the image file')
    if len(sys.argv)==1:
        parser.print_help()
        parser.exit()
    args = parser.parse_args()
    image_name = args.input_image
    assert image_name != None , "Why you are not putting input_image in argument?"

    print("Wait ...\n")

    img = cv2.imread(image_name)
    # img=cv2.resize(img, (800, 600))
    canny_edges = canny_edge_detector(img)

    h,w = canny_edges.shape[:2]

    try:
        print("Trying 1st MASK ...")
        mask = get_mask(h,w,0.8,1.0,0.5)
        copy=np.zeros_like(mask)
        for i in range(h):
            for j in range(w):
                if(mask[i][j]==255):
                    copy[i][j]=canny_edges[i][j]

        lines=get_lines(copy.astype(np.uint8))
        extrapolated_lines = extrapolate_line (h,w, lines)

    except (IndexError,TypeError):

        try:
            print("Trying 2nd MASK ...")
            mask = get_mask(h,w,0.9,1.0,0.5)
            copy=np.zeros_like(mask)
            for i in range(h):
                for j in range(w):
                    if(mask[i][j]==255):
                        copy[i][j]=canny_edges[i][j]

            lines=get_lines(copy.astype(np.uint8))
            extrapolated_lines = extrapolate_line (h,w, lines)

        except (IndexError,TypeError):

            try:
                print("Trying 3rd MASK ...")
                mask = get_mask(h,w,1.0,1.0,0.5)
                copy=np.zeros_like(mask)
                for i in range(h):
                    for j in range(w):
                        if(mask[i][j]==255):
                            copy[i][j]=canny_edges[i][j]

                lines=get_lines(copy.astype(np.uint8))
                extrapolated_lines = extrapolate_line (h,w, lines)

            except (IndexError,TypeError):

                try:
                    print("Trying 4th MASK ...")
                    mask = get_mask(h,w,1.0,0.7,0.5)
                    copy=np.zeros_like(mask)
                    for i in range(h):
                        for j in range(w):
                            if(mask[i][j]==255):
                                copy[i][j]=canny_edges[i][j]

                    lines=get_lines(copy.astype(np.uint8))
                    extrapolated_lines = extrapolate_line (h,w, lines)

                except (IndexError,TypeError):
                    pass
                    print("ERROR : Either No lane is there or lane is too small to get detected.")
                    exit()

    print("\n")
    output_img = draw_lane_on_image(img, extrapolated_lines, 5)
    # cv2.imwrite("canny.jpg",canny_edges)
    # cv2.imwrite("mask.jpg",mask)
    # cv2.imwrite("Masked_canny.jpg",copy)
    cv2.imwrite("output.jpg",output_img)

    # cv2.imshow("Lane detected",output_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print('Output has also been stored in `output.jpg` image file.\n')
    print("Have a nice day 😃\n")
