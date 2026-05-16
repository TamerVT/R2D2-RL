from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_camera_identifier(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a live OpenCV feed from the wrist camera."
    )
    parser.add_argument(
        "--camera-index-or-path",
        type=parse_camera_identifier,
        default="/dev/video0",
        help="Camera index or Linux device path, e.g. 0 or /dev/video0.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", type=str, default="MJPG")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "live_camera_snapshots",
    )
    return parser.parse_args()


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(args.camera_index_or_path, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera_index_or_path!r}."
        )

    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    return cap


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(args)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print("Live camera view opened.")
    print(f"  device:     {args.camera_index_or_path}")
    print(f"  requested:  {args.width}x{args.height} @ {args.fps} fps, {args.fourcc}")
    print(f"  actual:     {actual_width}x{actual_height} @ {actual_fps:.1f} fps")
    print("")
    print("Controls:")
    print("  q = quit")
    print("  s = save current frame")

    window_name = "SO101 Wrist Camera Live View"

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                print("Warning: failed to read frame.")
                continue

            display = frame_bgr.copy()
            cv2.putText(
                display,
                "q: quit | s: save snapshot",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = args.output_dir / f"camera_snapshot_{timestamp}.png"
                cv2.imwrite(str(path), frame_bgr)
                print(f"Saved snapshot: {path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
