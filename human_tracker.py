import cv2
import numpy as np
import time
import os
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

def select_video():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Video for Human Tracking & Counting",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )
    return file_path

def main():
    video_path = select_video()
    if not video_path:
        print("[ERROR] No video selected.")
        return

    output_dir = "Human_Analytics_Outputs"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(video_path)
    save_path = os.path.join(output_dir, f"Human_Count_{filename}")

    print("[INFO] Loading YOLOv8 Tracking Model...")
    # YOLOv8 nano model for fast real-time tracking
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    cv2.namedWindow("MALL FOOTFALL & HUMAN ANALYTICS ENGINE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MALL FOOTFALL & HUMAN ANALYTICS ENGINE", 1280, 720)

    # Define Counting Line (Horizontal line in the middle of screen)
    line_y = int(height * 0.55)
    crossed_ids = set()
    total_human_count = 0

    prev_time = time.time()

    print("[INFO] Processing Footfall & Human Tracking...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        fps_display = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else fps
        prev_time = curr_time

        # Run YOLOv8 Object Tracking with ByteTrack (persist=True for ID consistency)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

        # Draw Entry/Exit Virtual Counting Line (Yellow Line)
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 3)
        cv2.putText(frame, "VIRTUAL COUNTING GATE", (20, line_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        active_humans = 0

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()

            for box, track_id, cls_id in zip(boxes, track_ids, clss):
                # Filter only 'person' class (Class ID 0 in COCO Dataset)
                if int(cls_id) == 0:
                    active_humans += 1
                    x1, y1, x2, y2 = map(int, box)
                    track_id = int(track_id)

                    # Calculate Center Point of Person (Foot/Base level)
                    cx, cy = (x1 + x2) // 2, y2

                    # Distance Estimation Heuristic based on Bounding Box Height
                    box_height = y2 - y1
                    approx_distance = round(150.0 / (box_height + 1e-5), 1)

                    # Line Crossing Detection Logic
                    if abs(cy - line_y) < 15 and track_id not in crossed_ids:
                        crossed_ids.add(track_id)
                        total_human_count += 1

                    # Visual Styling: Change box color if crossed
                    box_color = (0, 200, 255) if track_id in crossed_ids else (0, 255, 0)

                    # Draw Bounding Box & Center Dot
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                    # ID & Distance Tag
                    tag = f"ID:{track_id} | {approx_distance}m"
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(tag)*9, y1), box_color, -1)
                    cv2.putText(frame, tag, (x1 + 5, y1 - 7), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

        # Tech Analytics HUD Panel
        cv2.rectangle(frame, (20, 20), (480, 120), (15, 15, 15), -1)
        cv2.putText(frame, "RETAIL FOOTFALL ANALYTICS", (30, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"CURRENT IN-FRAME : {active_humans} Persons", (30, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"TOTAL PASSED GATE : {total_human_count}", (30, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 170), 2)

        out.write(frame)
        cv2.imshow("MALL FOOTFALL & HUMAN ANALYTICS ENGINE", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Processed Output Saved at: {save_path}")

if __name__ == "__main__":
    main()