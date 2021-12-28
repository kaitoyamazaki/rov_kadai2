import cv2
import numpy as np

def main():

    img = cv2.imread('ball_kyougidai.jpg')
    
    while(1):
        #kernel = np.ones((5,5), np.float32)/ 25
        #img = cv2.filter2D(img, -1, kernel)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        low_green = np.array([100, 114, 60])
        upper_green = np.array([155, 255, 255])

        mask = cv2.inRange(hsv, low_green, upper_green)

        masking = cv2.bitwise_and(img, img, mask = mask)


        cv2.imshow('test1', img)
        cv2.imshow('test2', masking)

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
