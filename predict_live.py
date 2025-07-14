import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("gesture_model.h5")

# Define image size (must match training input shape)
IMG_SIZE = 64

# Define class labels (update if needed)
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect (optional)
    frame = cv2.flip(frame, 1)

    # Draw a rectangle where hand should be shown
    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # Extract the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=0)

    # Predict
    pred = model.predict(roi)
    label = labels[np.argmax(pred)]

    # Display prediction
    cv2.putText(frame, f"Predicted: {label}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
