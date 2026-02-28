import cv2

# Load an image from your "faces" folder
img = cv2.imread("faces/harshal1.jpg", cv2.IMREAD_GRAYSCALE)

# Make sure image is loaded
if img is None:
    print("❌ Could not read the image. Check the path and filename.")
    exit()

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Detect faces
faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

print("✅ Faces found:", len(faces))

# Optional: show image with detected face(s)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), 255, 2)

cv2.imshow("Face Detection Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
