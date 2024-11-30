# Sign Language Detector using Machine Learning

![Sign Language Detector](https://via.placeholder.com/800x200.png?text=Sign+Language+Detector)

You can download my training dataset here to work with the model: [Training Dataset](https://drive.google.com/drive/folders/1scsw0boul57btt2p0C4hlWGMyIPHrxuj?usp=sharing)

A real-time hand gesture recognition project to detect sign language letters using a webcam and a deep learning model. This project leverages the power of OpenCV, TensorFlow, and `cvzone` to collect, train, and predict hand gestures. The project aims to make communication easier for the hearing impaired by recognizing different sign language gestures.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Data Collection](#data-collection)
- [Model Training](#model-training)
- [Testing](#testing)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to recognize basic sign language letters through a webcam feed using machine learning. We use a Convolutional Neural Network (CNN) to classify images of hand gestures, which are captured in real-time. The captured images are processed, resized, and fed into a trained deep learning model that predicts the corresponding letter.

## Features

- Real-time hand gesture detection using a webcam.
- User-friendly interface using OpenCV.
- Train your own model using custom collected hand gesture data.
- Supports dynamic hand gesture recognition.

## How It Works

1. **Hand Detection**: Uses the `cvzone` Hand Tracking Module to detect hands in the video feed.
2. **Image Preprocessing**: Captures and processes images to match the input size expected by the model.
3. **Prediction**: Uses a trained CNN model to predict the sign language letter.
4. **Display**: Displays the predicted letter on the video feed.

The workflow consists of three main scripts: `datacollection.py`, `train_model.py`, and `test.py`.

### Data Collection (`datacollection.py`)

This script is used to collect training data by capturing hand gestures and saving them as images. It leverages OpenCV to capture frames from a webcam and uses `cvzone` to detect the hand.

```python
# Import necessary libraries
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Parameters
offset = 20
imgSize = 300
folder = "Data/A"
counter = 0

# Capture hand gestures
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Resize and place the cropped hand into a white image
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Save the image
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        counter += 1
        print(counter)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
```

### Model Training (`train_model.py`)

The next step is to train the collected images to recognize hand gestures. For training, we use a Convolutional Neural Network (CNN) model.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # Adjust the number of classes as needed
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    'Data',
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    'Data',
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=10)
model.save('Model/keras_model.h5')
```

### Testing (`test.py`)

This script uses the trained model to make predictions on real-time webcam input. It uses OpenCV to capture video frames, `cvzone` to detect the hand, and TensorFlow to classify the gesture.

```python
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Load pre-trained model and labels
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = ["A", "B", "C"]

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Make prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # Display prediction
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

## Installation

To run this project, follow these steps:

You can download my training dataset here to work with the model: [Training Dataset](https://drive.google.com/drive/folders/1scsw0boul57btt2p0C4hlWGMyIPHrxuj?usp=sharing)

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/sign-language-detector.git
   ```
2. **Install the required libraries**:
   ```sh
   pip install opencv-python tensorflow cvzone
   ```
3. **Run the data collection script** to capture gestures:
   ```sh
   python datacollection.py
   ```
4. **Train the model**:
   ```sh
   python train_model.py
   ```
5. **Run the testing script** to recognize gestures in real time:
   ```sh
   python test.py
   ```

## Results

The trained model can recognize sign language gestures in real time with reasonable accuracy. Below is a sample of the prediction:

![Gesture Recognition](https://via.placeholder.com/400x300.png?text=Prediction+Sample)

## Future Improvements

- Extend support for more letters and full words.
- Improve model accuracy by using more data and advanced architectures.
- Add a user interface for easier interaction.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
