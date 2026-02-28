# import cv2
# import os
# import numpy as np

# # Create LBPH recognizer and Haar cascade
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# faces = []
# labels = []
# label_map = {}

# # Create dataset folder if it doesn't exist
# os.makedirs("dataset", exist_ok=True)

# # Loop through images in 'faces' folder
# face_dir = "faces"
# for idx, filename in enumerate(os.listdir(face_dir)):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         path = os.path.join(face_dir, filename)
#         img = cv2.imread(path)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces_detected = detector.detectMultiScale(gray)

#         for (x, y, w, h) in faces_detected:
#             face_img = gray[y:y + h, x:x + w]
#             faces.append(face_img)
#             labels.append(idx)
#             label_map[idx] = os.path.splitext(filename)[0]

#             # Save cropped face to dataset/
#             cv2.imwrite(f"dataset/face_{idx}.jpg", face_img)

# # Save mapping for labels (optional)
# np.save("labels.npy", label_map)

# # Train recognizer
# if len(faces) >= 2:
#     recognizer.train(faces, np.array(labels))
#     recognizer.save("trainer.yml")
#     print("✅ Model trained and saved as trainer.yml")
# else:
#     print("❌ You need at least 2 images with detectable faces to train the model.")



import cv2
import numpy as np
import os
from PIL import Image

dataset_path = 'faces'
trainer_file = 'trainer.yml'
labels_file = 'labels.txt'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = []
labels = []
label_id = 0
label_map = {}

print("🧠 Starting training...")

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path) or person_name.startswith('.'):
        continue

    label_map[label_id] = person_name

    for img_name in os.listdir(person_path):
        if img_name.startswith('.'):
            continue

        img_path = os.path.join(person_path, img_name)
        try:
            img = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"❌ Skipping invalid image: {img_path} – {e}")
            continue

        img_np = np.array(img, 'uint8')
        detected_faces = detector.detectMultiScale(img_np)

        for (x, y, w, h) in detected_faces:
            face_region = img_np[y:y+h, x:x+w]
            faces.append(face_region)
            labels.append(label_id)

    label_id += 1

if len(faces) < 2:
    print("⚠️ Need at least 2 face samples to train. Add more images and try again.")
else:
    recognizer.train(faces, np.array(labels))
    recognizer.save(trainer_file)
    
    with open(labels_file, 'w') as f:
        for id_, name in label_map.items():
            f.write(f"{id_}:{name}\n")
    
    print(f"✅ Training complete. Model saved to {trainer_file}")
    print(f"✅ Labels saved to {labels_file}")
