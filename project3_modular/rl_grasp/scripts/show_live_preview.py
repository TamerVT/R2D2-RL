#!/usr/bin/env python3

from pathlib import Path
import time

import cv2


PREVIEW_FILE = Path("/tmp/lerobot_preview/wrist.jpg")


def main() -> None:
    print(f"Watching {PREVIEW_FILE}")
    print("Press q or Esc in the preview window to close it.")

    while True:
        if PREVIEW_FILE.exists():
            frame = cv2.imread(str(PREVIEW_FILE))
            if frame is not None:
                cv2.imshow("wrist", frame)

        key = cv2.waitKey(10) & 0xFF
        if key in (27, ord("q")):
            break

        time.sleep(0.01)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
