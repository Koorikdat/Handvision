import mediapipe as mp
import cv2 as cv
import time


# 1. Setup Paths and Options
model_path = '/Users/maisamanjum/Desktop/AI_Models/MediaPipeModels/gesture_recognizer.task'


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# 2. Define the Callback (This runs whenever the AI finds something)
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if result.gestures:
        for gesture in result.gestures:
            print(f"Gesture: {gesture[0].category_name} ({round(gesture[0].score, 2)})")


# 3. Initialize the Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

with GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe expects RGB, OpenCV gives BGR
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Send to recognizer with a millisecond timestamp
        frame_timestamp_ms = int(time.time() * 1000)
        recognizer.recognize_async(mp_image, frame_timestamp_ms)

        # Show the frame
        cv.imshow('MediaPipe Gesture Recognition', frame)
        
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()