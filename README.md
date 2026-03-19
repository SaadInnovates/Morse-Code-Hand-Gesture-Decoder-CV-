# Morse Code Hand Gesture Decoder

A real-time hand gesture recognition system that decodes Morse code using a webcam. The project leverages **OpenCV**, **CVZone**, and a custom **Keras deep learning model** to detect gestures and convert them into letters, numbers, and words.

---

## Features

- Detects 5 core gestures for Morse code:
  - `start` – undo last input
  - `space_letters` – separate letters
  - `space_words` – separate words
  - `dot` – Morse code dot (.)
  - `dash` – Morse code dash (-)
- Converts gestures to Morse code in real-time.
- Maps Morse code to **letters (A-Z), numbers (0-9)**.
- Real-time visualization with bounding boxes and current gesture display.
- Gesture stabilization using majority vote to reduce mispredictions.
- `start` gesture requires a hold to prevent accidental undo.


---

## Requirements

- Python 3.10+
- OpenCV (`opencv-python`)
- CVZone (`cvzone`)
- NumPy (`numpy`)
- TensorFlow/Keras (`tensorflow` / `keras`)
- Math, Time modules (Python built-in)

Install dependencies:

```bash
pip install opencv-python cvzone numpy tensorflow
```

---
### Folder Structure
Morse-Code-Hand-Gesture-Decoder-CV-/␣␣
│
├─ Model/
│   ├─ keras_model.h5          # Trained Keras model
│   └─ labels.txt              # Gesture labels
│
├─ main.py                     # Main script to run the program
├─ README.md
└─ .gitignore                  # Ignore venv, datasets, cache, etc.

---
### License
This project is open-source under the MIT License.
