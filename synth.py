import cv2
import numpy as np

def main():
    img = cv2.imread('ball_kyougidai.jpg')
    size = (640,480)

    cimg1 = img

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_blue = np.array([100, 100, 50])
    upper_blue = np.array([155, 255, 255])

    mask1 = cv2.inRange(hsv, low_blue, upper_blue)

    #masking = cv2.bitwise_and(img, img, mask = mask1)

    low_yellow = np.array([20, 105, 90])
    upper_yellow = np.array([50, 255, 255])

    mask2 = cv2.inRange(hsv, low_yellow, upper_yellow)

    #masking = cv2.bitwise_and(img, img, mask = mask2)
    low_red = np.array([165, 70, 70])
    upper_red = np.array([180, 255, 255])
    mask3 = cv2.inRange(hsv, low_red, upper_red)
    #masking = cv2.bitwise_and(img, img, mask = mask3)
    all_mask = mask1 + mask2 + mask3

    kernel = np.ones((5,5),np.uint8)

    all_mask = cv2.dilate(all_mask,kernel,iterations = 1)
    masking = cv2.bitwise_and(img, img, mask = all_mask)
    # Hough_tranceration
    #img = img[:,::-1]
    #masking = cv2.resize(masking, size)
    masking = cv2.GaussianBlur(masking, (33,33), 1)
    cimg2 = masking
    masking = cv2.cvtColor(masking, cv2.COLOR_RGB2GRAY)

    circles = cv2.HoughCircles(masking, cv2.HOUGH_GRADIENT, 1, 10,param1=50,param2=20,minRadius=5,maxRadius=30)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            #draw the outer circle
            cv2.circle(cimg1,(i[0],i[1]),i[2],(0,255,0),2)
            #draw the center of the circle
            cv2.circle(cimg1,(i[0],i[1]),2,(0,0,255),3)
    else:
        print("nothing")

    cv2.imshow('test1', cimg1)
    cv2.imshow('test2', all_mask)
    cv2.imshow('test3', cimg2)
        
    cv2.waitKey(0)

    #cap.release()
    #cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
