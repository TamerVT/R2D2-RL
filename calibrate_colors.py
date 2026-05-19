import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

"to detect run in the terminal python calibrate_colors.py --device"
"to save the detected colors to json file press S and to quit the instance Q"

MIN_PIXELS = 800  # minimum pixels in the largest blob to count as a real object


_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

COLOR_RANGES = [
    ("red",    [(0, 10), (170, 179)], 90,  60),
    ("orange", [(11, 22)],            130, 60),  # tighter hue + high saturation
    ("yellow", [(23, 38)],            90,  60),
    ("green",  [(39, 85)],            60,  50),
    ("blue",   [(86, 130)],           60,  50),
    ("purple", [(131, 160)],          80,  50),  # tighter hue + higher saturation
]

# BGR color for each label's dot/text overlay
LABEL_COLOR_BGR = {
    "red":    (0,   0,   220),
    "orange": (0,   140, 255),
    "yellow": (0,   220, 220),
    "green":  (0,   200, 0),
    "blue":   (220, 80,  0),
    "purple": (180, 0,   180),
}



Detection = tuple[str, tuple[int, int], int, int, int]  # label, centroid, r, g, b


def detect_all(frame_bgr: np.ndarray) -> list[Detection]:
    """Return one detection per color range that has enough pixels in the frame."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    fh, fw = frame_bgr.shape[:2]
    results: list[Detection] = []

    for label, hue_bands, s_min, v_min in COLOR_RANGES:
        mask = np.zeros((fh, fw), dtype=np.uint8)
        for h_lo, h_hi in hue_bands:
            m = cv2.inRange(hsv,
                            np.array([h_lo, s_min, v_min]),
                            np.array([h_hi, 255, 255]))
            mask = cv2.bitwise_or(mask, m)

        # remove noise: keep only solid blobs
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)

        # use only the largest connected blob
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n < 2:
            continue
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        if stats[largest, cv2.CC_STAT_AREA] < MIN_PIXELS:
            continue
        mask = np.uint8(labels == largest) * 255

        M = cv2.moments(mask)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cx = min(max(cx, 0), fw - 1)
        cy = min(max(cy, 0), fh - 1)

        b_val, g_val, r_val = frame_bgr[cy, cx]
        results.append((label, (cx, cy), int(r_val), int(g_val), int(b_val)))

    return results


def draw_frame(frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    vis = frame_bgr.copy()
    fw = vis.shape[1]

    for label, (cx, cy), _, _, _ in detections:
        dot_bgr = LABEL_COLOR_BGR.get(label, (255, 255, 255))

        # dot at centroid
        cv2.circle(vis, (cx, cy), 7, dot_bgr, -1)
        cv2.circle(vis, (cx, cy), 8, (0, 0, 0), 1)

        # label next to dot
        cv2.putText(vis, label, (cx + 10, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (cx + 10, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, dot_bgr, 1, cv2.LINE_AA)

    # bottom bar: list all detected colors
    if detections:
        summary = "  |  ".join(
            f"{lbl}  RGB({r},{g},{b})" for lbl, _, r, g, b in detections
        )
    else:
        summary = "no color detected"

    bar_y = vis.shape[0] - 30
    cv2.rectangle(vis, (0, bar_y - 20), (fw, vis.shape[0]), (30, 30, 30), -1)
    cv2.putText(vis, summary, (8, vis.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    return vis


def detections_to_dict(detections: list[Detection]) -> dict:
    return {label: {"r": r, "g": g, "b": b} for label, _, r, g, b in detections}


def save_dict(color_dict: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(color_dict, f, indent=4)
    print(f"Saved {len(color_dict)} colors to {path.resolve()}")
    print(color_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=str, default="detected_colors.json")
    args = parser.parse_args()

    output_path = Path(args.output)

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        sys.exit(f"[error] could not open camera device {args.device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Color detection running -- S to save dict  Q to quit")

    color_dict: dict = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        detections = detect_all(frame)
        color_dict = detections_to_dict(detections)
        vis = draw_frame(frame, detections)

        cv2.imshow("Color Detection -- S save  Q quit", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            save_dict(color_dict, output_path)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if color_dict:
        save_dict(color_dict, output_path)


if __name__ == "__main__":
    main()
