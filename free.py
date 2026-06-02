import cv2

cap = cv2.VideoCapture(0)  # 0 是默认摄像头

if not cap.isOpened():
    print("摄像头无法打开")
else:
    ret, frame = cap.read()
    if ret:
        print(f"摄像头正常工作，分辨率: {frame.shape}")
    else:
        print("摄像头无法读取画面")

    cap.release()