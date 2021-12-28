import cv2
import numpy as np

def main():

    img = cv2.imread('ball_kyougidai.jpg')
    
    while(1):
        #kernel = np.ones((5,5), np.float32)/ 25
        #img = cv2.filter2D(img, -1, kernel)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        low_yellow = np.array([20, 130, 100])
        upper_yellow = np.array([50, 255, 255])

        mask = cv2.inRange(hsv, low_yellow, upper_yellow)

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
