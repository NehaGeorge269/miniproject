import cv2
import numpy as np
import time

# Load YOLO-Tiny for better performance
weights_path = "D:\\ADMIN\\Downloads\\traffic\\traffic\\yolov4-tiny.weights"
config_path = "D:\\ADMIN\\Downloads\\traffic\\traffic\\yolov4-tiny.cfg"
net = cv2.dnn.readNet(weights_path, config_path)

# Get the output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
classes = open("D:\\ADMIN\\Downloads\\traffic\\traffic\\coco.names").read().strip().split("\n")
vehicle_classes = ["car", "motorbike", "bus", "truck"]

# Load video streams
video_paths = [
    "D:\\ADMIN\\Downloads\\traffic\\traffic\\video2.mp4",
    "D:\\ADMIN\\Downloads\\traffic\\traffic\\video1.mov",
    "D:\\ADMIN\\Downloads\\traffic\\traffic\\video3.mp4",
    "D:\\ADMIN\\Downloads\\traffic\\traffic\\video5.mp4"
]
caps = [cv2.VideoCapture(video) for video in video_paths]

# Traffic light position (right side)
light_x, light_y, light_size = 580, 20, 20

# Timer settings
MAX_TIME = 20
current_video = 0
start_time = time.time()

# Store last frames for each video to display when paused
last_frames = [None] * len(video_paths)

while True:
    frames = []
    vehicle_counts = []
    current_time = time.time()

    for i, cap in enumerate(caps):
        if i == current_video:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if it ends
                ret, frame = cap.read()
            last_frames[i] = frame
        else:
            frame = last_frames[i] if last_frames[i] is not None else np.zeros((360, 640, 3), dtype=np.uint8)

        # Resize frame for consistency
        frame = cv2.resize(frame, (640, 360))
        height, width = frame.shape[:2]

        vehicle_count = 0
        if i == current_video:
            # Convert to YOLO input format
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward(output_layers)

            for detection in detections:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5 and classes[class_id] in vehicle_classes:
                        vehicle_count += 1

                        # Get box coordinates (YOLO format gives center and width/height)
                        center_x = int(obj[0] * width)
                        center_y = int(obj[1] * height)
                        w = int(obj[2] * width)
                        h = int(obj[3] * height)

                        # Calculate top-left corner
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        # Draw a rectangle around the vehicle
                        color = (0, 255, 0)  # Green box for vehicles
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                        # Put label and confidence
                        label = f"{classes[class_id]}: {confidence:.2f}"
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # === Semi-transparent overlay ===
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (220, 100), (0, 0, 0), -1)  # Black box
            alpha = 0.5  # Transparency level
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Display vehicle count and timer inside the box
            elapsed_time = int(current_time - start_time)
            remaining_time = max(0, MAX_TIME - elapsed_time)
            cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Timer: {remaining_time}s", (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

        vehicle_counts.append(vehicle_count)

        # Traffic light logic:
        if i == current_video:
            if vehicle_count > 0:
                light_color = (0, 255, 0)  # Green if vehicle detected
            else:
                light_color = (0, 0, 255)  # Red if no vehicle detected
        else:
            light_color = (0, 0, 255)  # Red for non-active videos

        # Draw traffic light as small square on the right side
        cv2.rectangle(frame, (light_x, light_y), (light_x + light_size, light_y + light_size), light_color, -1)

        frames.append(frame)

    # Arrange frames in a 2x2 grid and display
    top_row = np.hstack((frames[0], frames[1]))
    bottom_row = np.hstack((frames[2], frames[3]))
    output_frame = np.vstack((top_row, bottom_row))

    cv2.imshow("Vehicle Detection with Traffic Lights", output_frame)

    # Switch to next video if time exceeds OR no vehicle found for 5 seconds
    if (current_time - start_time) >= MAX_TIME or (vehicle_counts[current_video] == 0 and (current_time - start_time) > 5):
        # Reset current video light to red
        last_frames[current_video] = frames[current_video].copy()
        cv2.rectangle(last_frames[current_video], (light_x, light_y), (light_x + light_size, light_y + light_size), (0, 0, 255), -1)

        # Move to the next video
        current_video = (current_video + 1) % len(video_paths)
        start_time = time.time()

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video resources
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
