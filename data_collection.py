import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Initialize hand detector (detect only 1 hand)
detector = HandDetector(maxHands=1)

# Padding around detected hand (to avoid tight cropping)
offset = 20

# Final output image size (square)
imgSize = 300

# Folder where images will be saved
folder = "Data/space_words" 

# Counter for saved images
counter = 0

while True:
    # Capture frame from webcam
    success, img = cap.read()

    # Detect hands and draw landmarks on image
    hands, img = detector.findHands(img)

    # If at least one hand is detected
    if hands:
        hand = hands[0]

        # Get bounding box of the hand
        x, y, w, h = hand['bbox']

        # Create a white background image (300x300)
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region with boundary safety (avoid negative indices)
        imgCrop = img[
            max(0, y - offset):min(img.shape[0], y + h + offset),
            max(0, x - offset):min(img.shape[1], x + w + offset)
        ]

        # Skip if crop is empty (can happen near edges)
        if imgCrop.size == 0:
            continue

        # Get shape of cropped image (not strictly needed, but useful for debugging)
        imgCropShape = imgCrop.shape

        # Calculate aspect ratio (height / width)
        aspectRatio = h / w

        # If hand is taller than wide
        if aspectRatio > 1:
            # Scale image to fit height
            k = imgSize / h
            wCal = math.ceil(k * w)

            # Resize while maintaining aspect ratio
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))

            # Calculate horizontal gap (to center image)
            wGap = math.ceil((imgSize - wCal) / 2)

            # Place resized image in the center horizontally
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            # Scale image to fit width
            k = imgSize / w
            hCal = math.ceil(k * h)

            # Resize while maintaining aspect ratio
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))

            # Calculate vertical gap (to center image)
            hGap = math.ceil((imgSize - hCal) / 2)

            # Place resized image in the center vertically
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Show cropped hand image
        cv2.imshow("ImageCrop", imgCrop)

        # Show processed (normalized) image
        cv2.imshow("ImageWhite", imgWhite)

    # Show original webcam feed
    cv2.imshow("Image", img)

    # Wait for key press (1 ms delay)
    key = cv2.waitKey(1)

    # If 's' key is pressed AND a hand is detected → save image
    if key == ord("s") and hands:
        counter += 1

        # Save image with unique timestamp filename
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)

        # Print number of saved images
        print(counter)