
import cv2

face_cascade = cv2.CascadeClassifier("resource/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("resource/haarcascade_eye.xml")

def face_detection(gray, original):
    face = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in face:
        cv2.rectangle(original, (x, y), (x + w, y + h), (255, 255, 255), 3)
        crop_vid = original[y:y + h, x:x + w]
        crop_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(crop_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(crop_vid, (ex, ey), (ex + ew, ey + eh), (255, 255, 255), 3)
    return original


web = cv2.VideoCapture(0)
while True:
    success, vid = web.read()
    vid_gray = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    detect = face_detection(vid_gray, vid)
    cv2.imshow("webcam",detect)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

web.release()
cv2.destroyAllWindows()