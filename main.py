import cv2
import numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image

# Initialize MediaPipe Hands module
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Detection Hand
def getHandMove(hand_landmarks):
    landmarks = hand_landmarks.landmark
    if all([landmarks[i].y < landmarks[i+3].y for i in range(9,20,4)]): 
        return "rock"
    elif landmarks[13].y < landmarks[16].y and landmarks[17].y < landmarks[20].y: 
        return "scissors"
    else: 
        return "paper"
    

# Set Height and Width Frame
frame_width = 1280
frame_height = 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Main Background
bg_image = cv2.imread("mainbackground.png")

# Use Font
font_path = 'VCR_OSD_MONO_1.001.ttf'
font = ImageFont.truetype(font_path, size = 25)

# Initialize Move Player and Text
p1_move = p2_move = None
gameText = ""
gameText1 = ""
gameText2 = ""
clock = 0 
hands_detected = False

# Set up parameters for hand detection
with mp_hands.Hands(model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
   
        if not ret or frame is None:
            continue

        # Convert the frame to RGB format for MediaPipe Hands module
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        result = hands.process(frame)

        # Convert the frame back to BGR format
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Initialize variables to store hand landmarks and background image
        player1_hand = None
        player2_hand = None
        bg_image_resized = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))

        # Create a mask for drawing rectangles
        mask = np.zeros_like(frame, dtype=np.uint8)

        # Check if hands are detected
        if result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # Draw landmarks and connections on the frame
                mp_drawing.draw_landmarks(frame,
                                          hand_landmarks,
                                          mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())
                

                # Define two rectangular boxes on the frame
                box_top_left_1 = (79, 300) 
                box_bottom_right_1 = (box_top_left_1[0] + 400, box_top_left_1[1] + 400)
                cv2.rectangle(mask, box_top_left_1, box_bottom_right_1, (255, 255, 255), thickness=cv2.FILLED)
                
                width = box_bottom_right_1[0] - box_top_left_1[0]

                box_top_left_2 = (frame_width - width - 79 - 28, 300) 
                box_bottom_right_2 = (frame_width - 107, box_top_left_2[1] + 400)
                cv2.rectangle(mask, box_top_left_2, box_bottom_right_2, (255, 255, 255), thickness=cv2.FILLED) 

                # Copy the content inside the boxes to the same location
                frame[box_top_left_1[1]:box_bottom_right_1[1], box_top_left_1[0]:box_bottom_right_1[0]] = frame[box_top_left_1[1]:box_bottom_right_1[1], box_top_left_1[0]:box_bottom_right_1[0]]
                frame[box_top_left_2[1]:box_bottom_right_2[1], box_top_left_2[0]:box_bottom_right_2[0]] = frame[box_top_left_2[1]:box_bottom_right_2[1], box_top_left_2[0]:box_bottom_right_2[0]]

                # Display the background in the area outside the boxes
                frame[~mask.astype(bool)] = bg_image_resized[~mask.astype(bool)]

                # Draw rectangles and store hand landmarks based on box location
                box_color = (255, 255, 255)  
                box_thickness = 2
                cv2.rectangle(frame, box_top_left_1, box_bottom_right_1, box_color, box_thickness)
                cv2.rectangle(frame, box_top_left_2, box_bottom_right_2, box_color, box_thickness)

                # Determine player based on box location
                if box_top_left_1[0] > frame.shape[1] / 2:
                    player1_hand = hand_landmarks
                else:
                    player2_hand = hand_landmarks
    
        # Perform some action using hand landmarks of player 1 and player 2
        if player1_hand and player2_hand:
            p1_move = getHandMove(player1_hand)
            p2_move = getHandMove(player2_hand)



        # If Multi Hand Detection Run this code
        if result.multi_hand_landmarks:
            # Set flag to indicate hands are detected
            hands_detected = True
            
            # Increment clock for timing purposes
            clock += 1

            # Timing logic to display countdown and start the game
            if 0 <= clock < 10:
                success = True
                gameText = "Ready?"
            elif clock < 15: gameText = "3..."
            elif clock < 30: gameText = "2..."
            elif clock < 45: gameText = "1..."
            elif clock < 60: gameText = "GO!!!"
            elif clock == 75:
                # Get hand moves for player 1 and player 2
                hls = result.multi_hand_landmarks
                if hls and len(hls) == 2:
                    p1_move = getHandMove(hls[0])
                    p2_move = getHandMove(hls[1])
                else:
                    success = False
            elif clock < 115:
                # Display game results
                if success:
                    # Determine winner based on hand moves
                    # Update gameText with the result
                    gameText1 = f"Player 1 : {p1_move}"
                    gameText2 = f"Player 2 : {p2_move}"
                    if p1_move == p2_move: gameText = f"Game is tied."
                    elif p1_move == "paper" and p2_move == "rock": gameText = f"Player 1 wins."
                    elif p1_move == "rock" and p2_move == "scissors": gameText = f"Player 1 wins."
                    elif p1_move == "scissors" and p2_move == "paper": gameText = f"Player 1 wins."
                    else: gameText = f"Player 2 wins."
                else:
                    gameText = "Didn't play properly!"
        else:
            # If no hands detected, reset variables
            hands_detected = False
            clock = 0
            
        # Convert frame to a PIL Image for drawing text
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        # Draw game information on the frame
        draw.text((100, 140), gameText1, font=font, fill=(255, 255, 255))
        draw.text((100, 175), gameText2, font=font, fill=(255, 255, 255))
        draw.text((100, 210), gameText, font=font, fill=(255, 255, 255))

        # Convert the frame back to OpenCV format
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)


        # If Not Multi Hand Detected
        if not result.multi_hand_landmarks:
            # Display a background image when no hands are detected
            bg_no_hand = cv2.imread('nohandbackground.jpg')
            frame = bg_no_hand

        # Show the final frame
        cv2.imshow('Games Rock Paper Scissors', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
