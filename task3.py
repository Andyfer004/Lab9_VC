import cv2
import time
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
SOURCE = "video2.mp4"
CONF = 0.45
IOU = 0.45
WINDOW_NAME = "Lab 9 - Task 3"
SAVE_OUTPUT = True
OUTPUT_PATH = "output_detection.mp4"

def draw_detections(frame, result, names):
    boxes = result.boxes
    count = 0

    if boxes is None or len(boxes) == 0:
        return frame, count

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[cls_id]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        count += 1

    return frame, count

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(SOURCE)

    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara o el video.")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0:
        fps_video = 30.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_video, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        results = model(frame, conf=CONF, iou=IOU, verbose=False)
        result = results[0]

        frame, detections = draw_detections(frame, result, model.names)

        end = time.time()
        fps = 1 / (end - start) if end > start else 0

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Detections: {detections}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"conf={CONF} iou={IOU}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 255, 200),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, frame)

        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()