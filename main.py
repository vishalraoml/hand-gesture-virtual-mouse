import cv2
import mediapipe as mp
import pyautogui
import math
import time


smoothening = 7
click_delay = 0.7


prev_x, prev_y = 0, 0
last_click_time = 0

# Screen size
screen_w, screen_h = pyautogui.size()

# Optional safety off
pyautogui.FAILSAFE = False

# CAMERA
cap = cv2.VideoCapture(0)

# MEDIAPIPE HANDS
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# MAIN LOOP
while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    result = hands.process(rgb)

    # Frame size
    h, w, c = frame.shape


    # HAND DETECTION
    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

        
            # INDEX FINGER
            index_x = int(hand.landmark[8].x * w)
            index_y = int(hand.landmark[8].y * h)

            cv2.circle(frame, (index_x, index_y), 10, (255, 0, 0), -1)

            
            # SCREEN COORDINATES
            mouse_x = int(index_x * screen_w / w)
            mouse_y = int(index_y * screen_h / h)

            # Prevent overflow
            mouse_x = max(0, min(screen_w, mouse_x))
            mouse_y = max(0, min(screen_h, mouse_y))

            # SMOOTH MOVEMENT
            curr_x = prev_x + (mouse_x - prev_x) / smoothening
            curr_y = prev_y + (mouse_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

            
            # THUMB
            thumb_x = int(hand.landmark[4].x * w)
            thumb_y = int(hand.landmark[4].y * h)

            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 0), -1)

            
            # CLICK DISTANCE
            click_distance = math.hypot(
                index_x - thumb_x,
                index_y - thumb_y
            )

            # Show click distance
            cv2.putText(
                frame,
                f"Click Dist: {int(click_distance)}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        
            # LEFT CLICK
            current_time = time.time()

            if click_distance < 50 and (current_time - last_click_time) > click_delay:

                pyautogui.click()

                last_click_time = current_time

                cv2.putText(
                    frame,
                    "LEFT CLICK",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

        
            # MIDDLE FINGER
            middle_x = int(hand.landmark[12].x * w)
            middle_y = int(hand.landmark[12].y * h)

            cv2.circle(frame, (middle_x, middle_y), 10, (255, 255, 0), -1)

           
            # SCROLL DISTANCE
            scroll_distance = math.hypot(
                index_x - middle_x,
                index_y - middle_y
            )

            # Show scroll distance
            cv2.putText(
                frame,
                f"Scroll Dist: {int(scroll_distance)}",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

           
            # SCROLL
            if scroll_distance > 80:

                # Scroll up
                if middle_y < index_y:
                    pyautogui.scroll(30)

                # Scroll down
                else:
                    pyautogui.scroll(-30)

                cv2.putText(
                    frame,
                    "SCROLL",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    3
                )

    # SHOW FRAME
    cv2.imshow("Gesture Mouse", frame)

    # ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break


# CLEANUP
cap.release()
cv2.destroyAllWindows()