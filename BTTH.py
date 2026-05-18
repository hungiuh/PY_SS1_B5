import cv2
import mediapipe as mp
import numpy as np
import time
# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
# Khởi tạo các biến
canvas = None
prev_x, prev_y = 0, 0
color = (0, 0, 255)  # Màu mặc định (đỏ)
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
color_index = 0
drawing = False
show_palette = False
color_change_cooldown = 0

# Biến tính FPS
fps_counter = 0
fps = 0
start_time = time.time()

# Mở camera
cap = cv2.VideoCapture(0)

def is_thumb_up(landmarks, w, h):
    """Kiểm tra cử chỉ giơ ngón cái lên"""
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]  # Khớp thứ 3 của ngón cái
    thumb_mcp = landmarks[2]  # Khớp thứ 2 của ngón cái
    wrist = landmarks[0]      # Cổ tay
    
    # Ngón cái phải cao hơn khớp thứ 2 và thứ 3
    thumb_up_vertical = thumb_tip[1] < thumb_ip[1] and thumb_tip[1] < thumb_mcp[1]
    
    # Ngón cái phải cách xa các ngón khác
    index_mcp = landmarks[5]
    thumb_away = thumb_tip[0] < index_mcp[0] - 30  # Ngón cái cách xa ngón trỏ
    
    return thumb_up_vertical and thumb_away

def count_fingers(landmarks):
    #Đếm số ngón tay đang giơ lên
    fingers = []
    
    # Ngón trỏ
    if landmarks[8][1] < landmarks[6][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Ngón giữa
    if landmarks[12][1] < landmarks[10][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Ngón áp út
    if landmarks[16][1] < landmarks[14][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Ngón út
    if landmarks[20][1] < landmarks[18][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    
    return sum(fingers)

def is_index_thumb_open(landmarks):
    """Kiểm tra ngón trỏ và ngón cái có mở ra không"""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    
    # Tính khoảng cách giữa ngón trỏ và ngón cái
    distance = np.sqrt((index_tip[0] - thumb_tip[0])**2 + (index_tip[1] - thumb_tip[1])**2)
    
    # Khoảng cách đủ xa để xác định là mở
    return distance > 80

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Tính FPS
    fps_counter += 1
    current_time = time.time()
    if current_time - start_time >= 1.0:
        fps = fps_counter
        fps_counter = 0
        start_time = current_time
        
    # Lật frame để có hiệu ứng gương
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Khởi tạo canvas nếu chưa có
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Giảm cooldown
    if color_change_cooldown > 0:
        color_change_cooldown -= 1
    
    # Chuyển đổi BGR sang RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Nhận diện tay
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Vẽ landmarks lên frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Lấy tọa độ các điểm quan trọng
            landmarks = []
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append((x, y))
            
            # Lấy tọa độ ngón trỏ và ngón cái
            index_tip = landmarks[8]  # Đầu ngón trỏ
            thumb_tip = landmarks[4]  # Đầu ngón cái
            
            # Kiểm tra cử chỉ
            thumb_up = is_thumb_up(landmarks, w, h)
            fingers_up = count_fingers(landmarks)
            index_thumb_open = is_index_thumb_open(landmarks)
            
            # Xử lý cử chỉ đổi màu (giơ ngón cái)
            if thumb_up and fingers_up == 0 and color_change_cooldown == 0:
                color_index = (color_index + 1) % len(colors)
                color = colors[color_index]
                color_change_cooldown = 30  # Cooldown để tránh đổi màu liên tục
                cv2.putText(frame, f"Color Changed!", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Hiển thị bảng màu khi mở bàn tay
            if fingers_up >= 4:
                show_palette = True
                cv2.putText(frame, "Color Palette: ON", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                show_palette = False
            
            # Vẽ bằng ngón trỏ (chỉ khi ngón trỏ và ngón cái mở ra)
            if index_thumb_open and fingers_up <= 1:  # Chỉ có ngón trỏ và ngón cái mở
                drawing = True
                if prev_x != 0 and prev_y != 0:
                    cv2.line(canvas, (prev_x, prev_y), index_tip, color, 5)
                prev_x, prev_y = index_tip
                # Hiển thị trạng thái vẽ
                cv2.putText(frame, "DRAWING", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                drawing = False
                prev_x, prev_y = 0, 0
           
            # Xóa canvas khi nắm tay
            if fingers_up == 0 and not thumb_up:  # Nắm tay
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(frame, "Canvas Cleared", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Hiển thị bảng màu nếu được yêu cầu
    if show_palette:
        palette_height = 50
        color_width = w // len(colors)
        for i, c in enumerate(colors):
            cv2.rectangle(frame, (i * color_width, 0), 
                         ((i + 1) * color_width, palette_height), c, -1)
            if i == color_index:
                cv2.rectangle(frame, (i * color_width, 0), 
                             ((i + 1) * color_width, palette_height), (255, 255, 255), 3)
    
    # Hiển thị thông tin
    cv2.putText(frame, f"FPS: {fps}", (w - 100, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Color: {color}", (10, h - 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)  
    # Kết hợp frame và canvas
    combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    
    # Hiển thị kết quả
    cv2.imshow('AirDraw Vision - Thumb Up to Change Color', combined)
    
    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()