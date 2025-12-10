from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
import io
import cv2
import numpy as np
from ultralytics import YOLO
import os

model = YOLO("best.pt")
app = FastAPI()

def detect_and_draw(frame):
    results = model(frame)
    detections = []

    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls)]
            conf = float(box.conf)
            print(f"Class: {cls}, Confidence: {conf:.2f}")
            # Draw bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            label = f"{cls} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            y_label = y1 + h + 10 if y1 + h + 10 < y2 else y2
            cv2.rectangle(frame, (x1, y1), (x1 + w, y_label), (0,255,0), -1)
            cv2.putText(frame, label, (x1, y1 + h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

            detections.append({
                "class": cls,
                "confidence": round(conf, 2),
                "bbox": [x1, y1, x2, y2]
            })

    return detections, frame

# -------------------- IMAGE UPLOAD --------------------
@app.post("/detect-img")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    detections, img_with_boxes = detect_and_draw(img)
    _, buffer = cv2.imencode(".png", img_with_boxes)
    io_buf = io.BytesIO(buffer)

    return StreamingResponse(io_buf, media_type="image/png")

# -------------------- VIDEO UPLOAD --------------------
@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    contents = await file.read()
    tmp_input = "temp_input.mp4"
    tmp_output = "temp_output.mp4"

    with open(tmp_input, "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture(tmp_input)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmp_output, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, frame_with_boxes = detect_and_draw(frame)
        out.write(frame_with_boxes)

    cap.release()
    out.release()

    def iterfile():
        with open(tmp_output, "rb") as f:
            yield from f

    return StreamingResponse(iterfile(), media_type="video/mp4")

# -------------------- LIVE WEBCAM STREAM --------------------

def generate_frames():
    cap = cv2.VideoCapture(0)  # Use default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, frame_with_boxes = detect_and_draw(frame)
        _, buffer = cv2.imencode('.jpg', frame_with_boxes)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video-feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

# -------------------- HELLO --------------------
@app.get("/hello")
def hello():
    return {"message": "Hello FastAPI"}

# uvicorn main:app --reload
# http://127.0.0.1:8000/docs