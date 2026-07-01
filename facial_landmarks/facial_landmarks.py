import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
RunningMode = vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=RunningMode.VIDEO,
)

with FaceLandmarker.create_from_options(options) as landmarker:

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        #Convert BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Create MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        #Correct timestamp (manual counter is safer)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.face_landmarks:
            print("Face Landmarks Detected")
        

        #Display
        cv2.imshow("Face Landmarker", cv2.flip(frame, 1))

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()