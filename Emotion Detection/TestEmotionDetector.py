import cv2
import numpy as np
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, InputLayer

# Emotion label dictionary
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

# Load model structure
with open(r'C:\Users\shubh\Desktop\MEGA PROJECT\Emotion detection\emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# ✅ Provide custom_objects to avoid Sequential deserialization error
custom_objects = {
    'Sequential': Sequential,
    'Conv2D': Conv2D,
    'MaxPooling2D': MaxPooling2D,
    'Dropout': Dropout,
    'Flatten': Flatten,
    'Dense': Dense,
    'InputLayer': InputLayer
}

emotion_model = model_from_json(loaded_model_json, custom_objects=custom_objects)
emotion_model.load_weights(r'C:\Users\shubh\Desktop\MEGA PROJECT\Emotion detection\emotion_model.h5')
print("✅ Model loaded from disk")

# Use webcam or video
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r'C:\Users\shubh\Desktop\MEGA PROJECT\Emotion detection\4587756-uhd_3840_2160_25fps.mp4')

# Load Haar cascade
face_detector = cv2.CascadeClassifier(
    r'C:\Users\shubh\Desktop\MEGA PROJECT\Emotion detection\haarcascade_frontalface_default.xml'
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray = gray_frame[y:y + h, x:x + w]
        if roi_gray.size == 0:
            continue
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cropped_img = cropped_img.astype("float32") / 255.0  # normalize input

        # Predict emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
