import cv2
import numpy as np

def pick_up_blue_ball():
    cap = cv2.VideoCapture(0) #カメラの指定
    #取得した画像のサイズ変換
    cap.set(3, 640)
    cap.set(4, 480)

    #無限ループ
    while True:

        #画像読み込み
        ret,img = cap.read()
        #画像のサイズを保管
        size = (640,480)
        #原画像保管
        cimg1 = img

        #原画像をhsvに変換 
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #下界を設定
        lower_red = np.array([160, 100, 100])

        #上界を設定
        upper_red = np.array([180, 255, 255])

        #マスク作成
        img_mask_red = cv2.inRange(hsv, lower_red, upper_red)

        #下界を設定
        lower_yellow = np.array([20, 80, 10])

        #上界を設定
        upper_yellow = np.array([50, 255, 255])

        #マスク作成
        img_mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        #下界を設定
        lower_blue = np.array([100, 100, 50])

        #上界を設定
        upper_blue = np.array([155, 255, 255])

        #マスク作成
        img_mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        all_mask = img_mask_red + img_mask_yellow + img_mask_blue

        #マスクと原画像の合成
        img_color_blue = cv2.bitwise_and(img, img, mask=all_mask)


        '''
        ここからハフ変換
        '''
        
        img = img[:,::-1] #ここでのimgは原画像
        
        '''
        現在の疑問点：個々で何をしているのか不明
        '''

        
        img_color_blue = cv2.resize(img_color_blue, size) #画像のリサイズ処理
        
        img_color_blue = cv2.GaussianBlur(img_color_blue, (33,33), 1) #ガウシアンフィルタを掛けているらしい？cv.GaussianBlur(img, (ax,ay), sigma_x(0の場合自動で指定される))

        cimg2 = img_color_blue# ここで画像の保存


        img_color_blue = cv2.cvtColor(img_color_blue, cv2.COLOR_RGB2GRAY) #画像をグレースケール化（ノイズ等を抽出しやすくするため）

        circles = cv2.HoughCircles(img_color_blue,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=10,maxRadius=120)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                #draw the outer circle
                cv2.circle(cimg1,(i[0],i[1]),i[2],(0,255,0),2)
                #draw the center of the circle
                cv2.circle(cimg1,(i[0],i[1]),2,(0,0,255),3)
        else:
            print("nothing")

        cv2.imshow('orizin',cimg1)
        cv2.imshow('blue',cimg2)
        k = cv2.waitKey(10)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    pick_up_blue_ball()


