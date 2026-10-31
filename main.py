""""
Управление:
    Выделение зоны: ЛКМ - добавить точку, R - сбросить, ENTER - готово (минимум 3 точки)
    Основное окно: Q или ESC - выход, Z - выбрать зону заново
"""

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"  
CONFIDENCE = 0.4         
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
CLASS_NAMES = {2:"car", 3:"motorcycle", 5:"bus", 7:"truck"}


def select_zone(frame):
    """Даёт пользователю кликами выделить зону (многоугольник) на кадре."""
    points = []
    window = "Select zone"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = frame.copy()
        for p in points:
            cv2.circle(canvas, p, 5, (0, 255, 255), -1)  
        if len(points) > 1:
            cv2.polylines(canvas, [np.array(points)], len(points) > 2, (0, 255, 255), 2)
        cv2.putText(canvas, "LMB: add point | R: reset | ENTER: done", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            points.clear()
        elif key == 13 and len(points) >= 3:  # Enter
            break
        elif key == 27:  # Esc
            cv2.destroyWindow(window)
            return None

    cv2.destroyWindow(window)
    return np.array(points, dtype=np.int32)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру/видео")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Не удалось получить кадр")

    zone = select_zone(frame)
    if zone is None:
        return

    model = YOLO(MODEL_PATH)
    seen_ids = set()  

    while True:
        ok, frame = cap.read()
        if not ok:
            break

       
        results = model.track(
            frame,
            persist=True,
            classes=VEHICLE_CLASSES,
            conf=CONFIDENCE,
            tracker="bytetrack.yaml",
            imgsz=960,
            verbose=False,
        )[0]

        in_zone_now = 0

        if results.boxes is not None:
            ids = results.boxes.id
            classes = results.boxes.cls.cpu().numpy().astype(int)
            for i, box in enumerate(results.boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)

           
                anchor = (int((x1 + x2) / 2), y2)
                inside = cv2.pointPolygonTest(zone, anchor, False) >= 0

                if inside:
                    in_zone_now += 1
                    if ids is not None:
                        seen_ids.add(int(ids[i]))

                color = (0, 255, 0) if inside else (200, 200, 200)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, anchor, 4, (0, 0, 255), -1)
                label = CLASS_NAMES[int(classes[i])]
                cv2.putText(frame, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

     
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone], (0, 0, 255) if in_zone_now else (0, 180, 0))
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        cv2.polylines(frame, [zone], True, (255, 255, 255), 2)

        cv2.putText(frame, f"In zone now: {in_zone_now}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        

        cv2.imshow("Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("z"):
            new_zone = select_zone(frame)
            if new_zone is not None:
                zone = new_zone
                seen_ids.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
