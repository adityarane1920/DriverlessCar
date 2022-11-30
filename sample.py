import cv2
import numpy as np
import RPi.GPIO as GPIO
import time


def run():
    print("Not detected")
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.HIGH)
    GPIO.output(in4,GPIO.LOW)


def wait(t):
    print("Stop")
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.LOW)
    time.sleep(t)



in1 = 24
in2 = 23
in3 = 16
in4 = 20
en1 = 25
en2 = 21
temp1=1

GPIO.setmode(GPIO.BCM)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(en1,GPIO.OUT)

GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
GPIO.setup(en2,GPIO.OUT)

GPIO.output(in1,GPIO.HIGH)
GPIO.output(in2,GPIO.LOW)

GPIO.output(in3,GPIO.HIGH)
GPIO.output(in4,GPIO.LOW)

p=GPIO.PWM(en1,1000)
q=GPIO.PWM(en2,1000)

p.start(20)
q.start(20)


stop_sign = cv2.CascadeClassifier('/home/pi/Desktop/cascade_stop_sign.xml')
zebrasign = cv2.CascadeClassifier('/home/pi/Desktop/zebra.xml')

cap = cv2.VideoCapture(0)

i=0

while cap.isOpened():
    run()
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hav =  cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    stop_sign_scaled = stop_sign.detectMultiScale(gray, 1.3, 5)
    zebra = zebrasign.detectMultiScale(gray, 1.3, 5)
    
    lower_blue = np.array([141,155,84])
    upper_blue = np.array([179,255,255])
    
    mask = cv2.inRange(hav,lower_blue,upper_blue)
    blue  = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    
    for (x, y, w, h) in zebra:
            # Draw rectangle around the stop sign
        zebra_rectangle = cv2.rectangle(img, (x,y),
                                            (x+w, y+h),
                                            (0, 255, 0), 3)
        # Write "Stop sign" on the bottom of the rectangle
        zebra_text = cv2.putText(img=zebra_rectangle,
                                     text="Zebra Crossing",
                                     org=(x, y+h+30),
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                     fontScale=1, color=(0, 0, 255),
                                     thickness=2, lineType=cv2.LINE_4)
    
    
        i=i+1
        if i==1:
            wait(6)
        else:
            run()            
    
        
    for (x, y, w, h) in stop_sign_scaled:
        # Draw rectangle around the stop sign
        stop_sign_rectangle = cv2.rectangle(img, (x,y),
                                            (x+w, y+h),
                                            (0, 255, 0), 3)
        # Write "Stop sign" on the bottom of the rectangle
        stop_sign_text = cv2.putText(img=stop_sign_rectangle, 
                                     text="Stop Sign",
                                     org=(x, y+h+30),
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                     fontScale=1, color=(0, 0, 255),
                                     thickness=2, lineType=cv2.LINE_4)
    
    
        wait(6)
        break
    
    if len(blue)>4:
            blue_area = max(blue,key=cv2.contourArea)
            (xg,yg,wg,hg)=cv2.boundingRect(blue_area)
            cv2.rectangle(img,(xg,yg),(xg+wg,yg+hg),(0,255,2),2)
            print("Object detected")
            wait(3)
    else: 
            run()
    
    
    
    cv2.imshow("img", img)
    key = cv2.waitKey(30)
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break



    
                                                                                                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                        


