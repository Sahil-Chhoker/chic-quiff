import cv2

# Path to the Haar cascade XML file
alg = r"C:\MASTER FOLDER\Portfolio-Projects\chic-quiff\Models\haarcascade_frontalface_default.xml"

# Load the cascade classifier
haar_cascade = cv2.CascadeClassifier(alg)

# Path to the image file
file_name = r"C:\MASTER FOLDER\Portfolio-Projects\chic-quiff\Images\sampleImg.png"

# Read the image
img = cv2.imread(file_name)

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = haar_cascade.detectMultiScale(
    gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100)
)

# Process each detected face
for i, (x, y, w, h) in enumerate(faces):
    # Crop the face region
    cropped_image = img[y:y+h, x:x+w]

    # Save the cropped face image
    target_file_name = f'stored-faces/{i}.jpg'
    cv2.imwrite(target_file_name, cropped_image)
