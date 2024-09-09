import cv2
import numpy as np
from collections import deque


buffer_size = 16
pts = deque(maxlen=buffer_size)


#mavi renk aralığı HSV
blueLower=(105,98,0)
blueUpper=(179,240,255)

#capture

cap = cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,480)

while True:
    success ,img = cap.read()
    
    if success:
        #blur
        blurred = cv2.GaussianBlur(img, ksize=(11,11), sigmaX=0)
        #hsv donustur
        hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
        
        #cv2.imshow("HSV IMG",hsv)
        
        #mavi icin maske olustur
        mask = cv2.inRange(hsv,blueLower,blueUpper)
        #cv2.imshow("mask",mask)
        
        #maskenin etrafında kalan gurultuleri sil
        
        mask = cv2.erode(mask,None,iterations=1)
        mask = cv2.dilate(mask,None, iterations=1)
        cv2.imshow("Masked + erezyon ve genişleme",mask)
        
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        center = None
        
        if len(contours) > 0:
            
            #en buyuk konturu al
            c = max(contours,key=cv2.contourArea)
            
            #dikdortgene cevir
            rect = cv2.minAreaRect(c)
            ((x,y),(witdh,height),rotation) = rect
            
            
            s = "x:{},y:{},width:{},height:{},rotation:{}".format(np.round(x),np.round(y),np.round(witdh),np.round(height),np.round(rotation))
            print(s)
            
            
            #tespit edilen nesneyi diktorgen icine al
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            
            #moment goruntunun merkezini bulmaya yarar
            M = cv2.moments(c)
            center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
                      
            #kontoru cizdir sarı
            cv2.drawContours(img,[box],0,(0,255,255),2)
            #merkeze bir tane nokta çizelim
            cv2.circle(img, center, 5, (255,0,255),-1)
            #bilgileri ekrana yazdır
            cv2.putText(img,s,(30,30),cv2.FONT_HERSHEY_COMPLEX,1,(0.0,0),1)            
           
            
            
            #deque 
            pts.appendleft(center)
            for i in range(1,len(pts)):
                if pts[i-1] is None or pts[i] is None: continue
                cv2.line(img, pts[i-1], pts[i],(0,255,0),9)
                cv2.imshow("Orijinal Tespit", img)
            else: cv2.imshow("Orijinal Tespit", img)
    else: cap.release(),cv2.destroyAllWindows()
    
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
        break