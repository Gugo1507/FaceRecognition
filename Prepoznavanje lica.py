import cv2
from deepface import DeepFace
image_path=input("Upišite poveznicu do slike ili stisnite Enter za live camera face recognition: ")
if image_path:
    image = cv2.imread(image_path)
    results = DeepFace.analyze(image, actions=['gender', 'age', 'emotion'])
    first_face = results[0]
    dominant_gender = first_face['dominant_gender']
    dominant_age = first_face['age']
    dominant_emotion = first_face['dominant_emotion']
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, str(dominant_emotion), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(image, str(dominant_age), (x, y - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(image, str(dominant_gender), (x, y - 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow('Camera Feed', image)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()

else:
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    results = DeepFace.analyze(frame, actions=['gender', 'age', 'emotion'])

    if len(results) > 0:
        first_face = results[0]
        dominant_gender = first_face['dominant_gender']
        dominant_age = first_face['age']
        dominant_emotion = first_face['dominant_emotion']
        cap.release()
        cv2.destroyAllWindows()


    else:
        print("Nema lica")

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, str(dominant_emotion), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(frame, str(dominant_age), (x, y - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(frame, str(dominant_gender), (x, y - 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow('Camera Feed', frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()












