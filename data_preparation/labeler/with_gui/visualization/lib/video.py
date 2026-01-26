import cv2
import numpy as np
from data_preparation.labeler.with_gui.visualization.lib.image import put_text, draw_matches

def play_trip(video_path, lat_lon=None, timestamps=None, color_mode=False, wait_time=100, win_name="Trip"):
    """
    Plays back a video file directly from disk (streaming) instead of a list.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # If the original code logic expected grayscale but color_mode is False
        if not color_mode:
            # Check if frame is already grayscale (1 channel)
            if len(frame.shape) == 3:
                show_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Convert back to BGR just for the colored text/overlays
                show_image = cv2.cvtColor(show_image, cv2.COLOR_GRAY2BGR)
            else:
                show_image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            show_image = frame

        # Overlay Info
        show_image = put_text(show_image, "top_left", "Press ESC to stop")
        show_image = put_text(show_image, "top_right", f"Frame: {frame_count}/{total_frames}")

        if timestamps is not None and frame_count < len(timestamps):
            show_image = put_text(show_image, "bottom_right", f"{timestamps[frame_count]}")

        if lat_lon is not None and frame_count < len(lat_lon):
            lat, lon = lat_lon[frame_count]
            show_image = put_text(show_image, "bottom_left", f"{lat}, {lon}")

        cv2.imshow(win_name, show_image)
        
        if cv2.waitKey(wait_time) == 27: # ESC key
            break
            
        frame_count += 1

    cap.release()
    cv2.destroyWindow(win_name)