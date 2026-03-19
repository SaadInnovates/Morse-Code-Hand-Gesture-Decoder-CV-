import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from collections import deque

# --- Setup ---
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
labels = ["start", "space_words", "space_letters", "dot", "dash"]  # gestures

# --- Extended Morse code dictionary ---
MORSE_CODE_DICT = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9"
}

# --- Variables ---
current_morse = ""
result_text = ""
last_append_time = 0
append_delay = 1.5  # reduced since we stabilize gestures
gesture_history = deque(maxlen=5)  # store last 5 predictions for majority vote
start_pressed_time = 0

# --- Main Loop ---
while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create white canvas
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop hand with offset
        imgCrop = img[
            max(0, y - offset):min(y + h + offset, img.shape[0]),
            max(0, x - offset):min(x + w + offset, img.shape[1])
        ]

        if imgCrop.size == 0:
            continue

        aspectRatio = h / w
        if aspectRatio > 1:  # Tall
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:  # Wide
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # --- Predict gesture ---
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        gesture_history.append(labels[index])
        # Majority vote for stabilization
        current_gesture = max(set(gesture_history), key=gesture_history.count)

        # --- Handle gestures ---
        current_time = time.time()
        
        # dot and dash
        if current_gesture in ["dot", "dash"] and current_time - last_append_time > append_delay:
            current_morse += "." if current_gesture == "dot" else "-"
            last_append_time = current_time

        # space between letters
        elif current_gesture == "space_letters" and current_time - last_append_time > append_delay:
            if current_morse in MORSE_CODE_DICT:
                result_text += MORSE_CODE_DICT[current_morse]
            current_morse = ""
            last_append_time = current_time

        # space between words
        elif current_gesture == "space_words" and current_time - last_append_time > append_delay:
            if current_morse in MORSE_CODE_DICT:
                result_text += MORSE_CODE_DICT[current_morse]
            result_text += "_"
            current_morse = ""
            last_append_time = current_time

        # start (undo) requires hold
        elif current_gesture == "start":
            if start_pressed_time == 0:
                start_pressed_time = time.time()
            elif time.time() - start_pressed_time > 1.0:  # hold for 1 second
                if current_morse:
                    current_morse = current_morse[:-1]
                elif result_text:
                    result_text = result_text[:-1]
                start_pressed_time = 0
                last_append_time = current_time
        else:
            start_pressed_time = 0  # reset if gesture changes

        # --- Draw gesture and bounding box ---
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 250, y - offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, f"Current: {current_gesture}", (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # --- Display full result in frame ---
    cv2.rectangle(imgOutput, (10, 10), (1200, 80), (0, 0, 0), cv2.FILLED)
    cv2.putText(imgOutput, f"Result: {result_text}{current_morse}", (20, 60),
                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Image", imgOutput)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()