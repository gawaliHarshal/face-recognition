# import cv2
# import numpy as np

# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read("trainer.yml")
# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# # Load label map
# label_map = np.load("labels.npy", allow_pickle=True).item()

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         face_id, conf = recognizer.predict(gray[y:y+h, x:x+w])

#         name = label_map.get(face_id, "Unknown")
#         color = (0, 255, 0) if conf < 70 else (0, 0, 255)
#         label = f"{name} ({int(conf)})" if conf < 70 else "Unknown"

#         cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#         cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     cv2.imshow("Face Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load label mapping
label_map = {}
with open("labels.txt", "r") as f:
    for line in f:
        id_str, name = line.strip().split(":")
        label_map[int(id_str)] = name

cap = cv2.VideoCapture(0)

print("📷 Starting camera. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
        name = label_map.get(id_, "Unknown")

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        label_text = f"{name} ({int(conf)}%)"
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
