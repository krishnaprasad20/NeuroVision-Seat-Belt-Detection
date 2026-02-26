import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("seatbelt_detector.h5")

class_labels = {0: "With Seatbelt", 1: "With out Seatbelt"}

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for prediction
    img = cv2.resize(frame, (224, 224))
    img_array = img / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    predicted_class = 1 if prediction > 0.6 else 0
    label = class_labels[predicted_class]
    confidence = prediction*100 if predicted_class == 1 else (1 - prediction)*100

    # Color based on result
    color = (0, 0, 255) if predicted_class == 1 else (0, 255, 0)

    # Display result
    cv2.putText(frame, f"{label} ({confidence:.1f}%)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Seatbelt Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()