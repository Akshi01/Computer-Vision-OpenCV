
import cv2
import numpy as np

img = cv2.imread('resource/f.jpg')
cv2.imshow("pic",img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_classifier = cv2.CascadeClassifier('resource/haarcascade_frontalface_default.xml')
eyes_classifier = cv2.CascadeClassifier('resource/haarcascade_eye.xml')

face = face_classifier.detectMultiScale(gray, 1.3, 5)

if face is None:
    print("No face detected")

for (x,y,w,h) in face:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 3)
    cv2.imshow("face", img)
    cv2.waitKey()
    crop_img = img[y:y + h, x:x + w]
    crop_gray = gray[y:y + h, x:x + w]
    eyes = eyes_classifier.detectMultiScale(crop_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(crop_img, (ex,ey), (ex+ew, ey+eh), (255,255,255), 3)
    cv2.imshow("",img)
    cv2.waitKey()

cv2.destroyAllWindows()