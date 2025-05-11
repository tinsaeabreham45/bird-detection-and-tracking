import gradio as gr
import cv2
import uuid
import os
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load model and tracker
model = YOLO("my_model.pt")
tracker = DeepSort(max_age=30)

# Process image
def process_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(img)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"Bird {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

    # Apply Deep SORT
    tracks = tracker.update_tracks(detections, frame=img)
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(img, f"ID {tid}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Process video
def process_video(video):
    return gr.update(value=None, visible=True), process_video_internal(video)

def process_video_internal(video):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0

    out_path = f"output_{uuid.uuid4().hex}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    tracking_data = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        resized = cv2.resize(frame, (640, 640))
        results = model(resized)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

        tracks = tracker.update_tracks(detections, frame=resized)
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cv2.rectangle(resized, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(resized, f"ID {tid}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if out is None:
            h, w, _ = resized.shape
            out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        out.write(resized)

    cap.release()
    out.release()

    return out_path

# UI
with gr.Blocks() as demo:
    gr.Markdown("## 🐦 Bird Detection & Tracking App")
    gr.Markdown("Upload an image or a video. The model will detect and track birds using YOLOv11 + Deep SORT.")

    with gr.Tab("Image"):
        image_input = gr.Image(type="pil", label="Upload an Image")
        image_output = gr.Image(type="numpy", label="Result Image")
        image_btn = gr.Button("Detect Birds")
        image_btn.click(fn=process_image, inputs=image_input, outputs=image_output)

    with gr.Tab("Video"):
        video_input = gr.Video(label="Upload a Video")
        wait_text = gr.Textbox(label="Status", visible=False)
        video_output = gr.Video(label="Result Video")
        video_btn = gr.Button("Track Birds")

        video_btn.click(fn=process_video, inputs=video_input, outputs=[wait_text, video_output])

demo.launch()
