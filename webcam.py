from collections import deque
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
RunningMode = vision.RunningMode

class NoddStages:
    """
    Detects head nod gestures (up and down) using facial landmarks.
    Uses a state machine to track the progression of a nod from idle -> moving away -> returning -> idle.
    """
    def __init__(self):
        self.state = "IDLE"  # IDLE, MOVING_AWAY, or RETURNING
        self.gestureDirection = None  # UP, DOWN, or None
        self.isUpNod = False
        self.isDownNod = False
        self.isLeftNod = False
        self.isRightNod = False
        self.prevState = None
        self.prev_average_y = None
        self.prev_average_x = None
        self.window_size = 5
        self.y_buffer = deque(maxlen=self.window_size)
        self.x_buffer = deque(maxlen=self.window_size)
        self.sensitivity_up_away = 0.01
        self.sensitivity_up_return = 0.01
        self.sensitivity_down_away = 0.01
        self.sensitivity_down_return = 0.01
        self.sensitivity_left_away = 0.01
        self.sensitivity_left_return = 0.01
        self.sensitivity_right_away = 0.01
        self.sensitivity_right_return = 0.01
        self.cooldown_ms = 800  # Prevent rapid nod re-detection
        self.last_nod_time = 0
        self.idle_cooldown_ms = 400 #If no movement is detected for this time, reset the state machine
        self.last_movement_time = float('inf')  # Initialize to negative infinity to ensure the first movement is detected
    def reset_tracking(self):
        """Reset the state machine and buffers to initial state."""
        self.state = "IDLE"
        self.gestureDirection = None
        self.prev_average_y = None
        self.prev_average_x = None
        self.y_buffer.clear()
        self.x_buffer.clear()
        self.isDownNod = False
        self.isUpNod = False
        self.isLeftNod = False
        self.isRightNod = False
        self.last_movement_time = float('inf')
    def parse_landmarks(self, face):
        """Extract facial landmarks and return smoothed position changes."""
        if not face:
            return None
        nose_tip = face[1]
        chin = face[152]
        left_eye = face[33]
        right_eye = face[263]
        left_mouth = face[61]
        right_mouth = face[291]
        forehead = face[10]
        
        average_y = (nose_tip.y + chin.y + left_eye.y + right_eye.y + left_mouth.y + right_mouth.y + forehead.y) / 7
        average_x = (nose_tip.x + chin.x + left_eye.x + right_eye.x + left_mouth.x + right_mouth.x + forehead.x) / 7
        
        self.y_buffer.append(average_y)
        self.x_buffer.append(average_x)
        
        smooth_y = sum(self.y_buffer) / len(self.y_buffer)
        smooth_x = sum(self.x_buffer) / len(self.x_buffer)

        delta_y = smooth_y - self.prev_average_y if self.prev_average_y is not None else 0
        delta_x = smooth_x - self.prev_average_x if self.prev_average_x is not None else 0
        return delta_y, delta_x, smooth_y, smooth_x
    
    def update(self, face, timestamp_ms):
        """Update nod detection state machine. States: IDLE -> MOVING_AWAY -> RETURNING -> IDLE."""
        if timestamp_ms - self.last_nod_time < self.cooldown_ms:
            self.reset_tracking()
            return
        if timestamp_ms - self.last_movement_time > self.idle_cooldown_ms:
            self.reset_tracking()
            return
        result = self.parse_landmarks(face)
        if not result:
            self.reset_tracking()
            return
        delta_y, delta_x, smooth_y, smooth_x = result

        if self.state == "IDLE":
            self.last_movement_time = timestamp_ms
            if delta_y > self.sensitivity_down_away:
                self.state = "MOVING_AWAY"
                self.gestureDirection = "DOWN"
                self.last_movement_time = timestamp_ms
                print("Moving down")
            elif delta_y < -self.sensitivity_up_away:
                self.state = "MOVING_AWAY"
                self.gestureDirection = "UP"
                self.last_movement_time = timestamp_ms
                print("Moving up")
            elif delta_x > self.sensitivity_right_away:
                self.state = "MOVING_AWAY"
                self.gestureDirection = "LEFT"
                self.last_movement_time = timestamp_ms
                print("Moving left")
            elif delta_x < -self.sensitivity_left_away:
                self.state = "MOVING_AWAY"
                self.gestureDirection = "RIGHT"
                self.last_movement_time = timestamp_ms
                print("Moving right")
        elif self.state == "MOVING_AWAY":
            if delta_y < -self.sensitivity_down_return and self.gestureDirection == "DOWN":
                self.state = "RETURNING"
                self.last_movement_time = timestamp_ms
            elif delta_y > self.sensitivity_up_return and self.gestureDirection == "UP":
                self.state = "RETURNING"
                self.last_movement_time = timestamp_ms
            elif delta_x < -self.sensitivity_right_return and self.gestureDirection == "LEFT":
                self.state = "RETURNING"
                self.last_movement_time = timestamp_ms
            elif delta_x > self.sensitivity_left_return and self.gestureDirection == "RIGHT":
                self.state = "RETURNING"
                self.last_movement_time = timestamp_ms
        elif self.state == "RETURNING":
            if self.gestureDirection == "DOWN" and delta_y < -self.sensitivity_down_return:
                self.state = "IDLE"
                self.isDownNod = True
                self.last_nod_time = timestamp_ms
                print("Down Nod detected!")
            elif self.gestureDirection == "UP" and delta_y > self.sensitivity_up_return:
                self.state = "IDLE"
                self.isUpNod = True
                self.last_nod_time = timestamp_ms
                print("Up Nod detected!")
            elif self.gestureDirection == "LEFT" and delta_x < -self.sensitivity_right_return:
                self.state = "IDLE"
                self.isLeftNod = True
                self.last_nod_time = timestamp_ms
                print("Left Nod detected!")
            elif self.gestureDirection == "RIGHT" and delta_x > self.sensitivity_left_return:
                self.state = "IDLE"
                self.isRightNod = True
                self.last_nod_time = timestamp_ms
                print("Right Nod detected!")
        self.prev_average_y = smooth_y
        self.prev_average_x = smooth_x
        

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="facial_landmarks/face_landmarker.task"),
    running_mode=RunningMode.VIDEO,
)
noddTracker = NoddStages()

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
        face = result.face_landmarks[0] if result.face_landmarks else None

        if face:
            noddTracker.update(face, timestamp_ms)
        #Display
        cv2.imshow("Face Landmarker", cv2.flip(frame, 1))

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()