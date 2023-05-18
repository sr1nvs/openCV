import dlib
import cv2
detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
        confidence = face.confidence
        strc = str(confidence)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(frame,strc,(x,y),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    cv2.imshow('balls',frame)
    if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
