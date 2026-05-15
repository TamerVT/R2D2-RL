import cv2
import numpy as np
import argparse



def _grid_object_points(grid_shape=(4, 4), spacing_mm=25.0):
    """Return (N,3) float32 object points on Z=0 plane in mm."""
    rows, cols = grid_shape
    obj = np.zeros((rows * cols, 3), dtype=np.float32)
    i = 0
    for r in range(rows):
        for c in range(cols):
            obj[i] = (c * spacing_mm, r * spacing_mm, 0.0)
            i += 1
    return obj


def _init_grid_image_points(frame_shape, grid_shape=(4, 4), margin=0.2):
    """Initial guess for image points as a regular grid inside a rectangle."""
    h, w = frame_shape[:2]
    rows, cols = grid_shape

    mx = int(margin * w)
    my = int(margin * h)

    tl = np.array([mx, my], dtype=np.float64)
    tr = np.array([w - mx, my], dtype=np.float64)
    br = np.array([w - mx, h - my], dtype=np.float64)
    bl = np.array([mx, h - my], dtype=np.float64)

    pts = []
    for r in range(rows):
        v = r / (rows - 1) if rows > 1 else 0.0
        left  = (1 - v) * tl + v * bl
        right = (1 - v) * tr + v * br
        for c in range(cols):
            u = c / (cols - 1) if cols > 1 else 0.0
            p = (1 - u) * left + u * right
            pts.append(p)
    return np.array(pts, dtype=np.float64)  # (N,2)


def run_grid_calibration_ui(
    cap,
    grid_shape=(4, 4),
    spacing_mm=25.0,
    n_views=8,
    win="calib",
    flags=None
):
    """
    Live calibration UI with N=rows*cols user-moveable points.
    ENTER accepts a view. Returns K, distCoeffs, rms, rvecs, tvecs, image_size.
    """
    if flags is None:
        # conservative defaults to reduce overfitting if you take few views
        flags = cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # First frame to get size
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Could not read from camera.")
    h, w = frame.shape[:2]

    rows, cols = grid_shape
    N = rows * cols

    # Object points (same for every view)
    objp = _grid_object_points(grid_shape, spacing_mm)  # (N,3) float32

    # Collected data (LIST OF ARRAYS, one per view) [2](https://github.com/filoucool/Python-Pick-and-Place/blob/master/detector.py)[1](https://www.geeksforgeeks.org/python/how-to-create-requirements-txt-file-in-python/)
    objpoints = []
    imgpoints = []

    # Movable image points (float64 while editing)
    pts = _init_grid_image_points(frame.shape, grid_shape, margin=0.2)  # (N,2)

    selected = 0
    step = 3
    frozen = False
    frozen_frame = None
    view_idx = 0

    def idx_to_rc(i):
        return (i // cols, i % cols)

    while True:
        # get frame
        if not frozen:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            display = frame.copy()
        else:
            display = frozen_frame.copy()

        # draw points + labels
        for i, p in enumerate(pts):
            r, c = idx_to_rc(i)
            color = (0, 255, 255) if i == selected else (0, 200, 0)
            cv2.circle(display, (int(p[0]), int(p[1])), 6, color, -1)
            cv2.putText(display, f"{r},{c}", (int(p[0]) + 7, int(p[1]) - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # HUD
        hud = [
            f"Grid {rows}x{cols} | View {view_idx}/{n_views} | selected={selected} rc={idx_to_rc(selected)} | step={step}px | frozen={frozen}",
            "Keys: [ ] prev/next | arrows/WASD move | +/- step | SPACE freeze | ENTER accept | BACKSPACE reset | Q/ESC quit"
        ]
        y0 = 25
        for line in hud:
            cv2.putText(display, line, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            y0 += 22

        cv2.imshow(win, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

        # freeze
        if key == ord(' '):
            frozen = not frozen
            if frozen:
                frozen_frame = frame.copy()

        # step size
        elif key in (ord('+'), ord('=')):
            step = min(50, step + 1)
        elif key in (ord('-'), ord('_')):
            step = max(1, step - 1)

        # select point
        elif key == ord('['):
            selected = (selected - 1) % N
        elif key == ord(']'):
            selected = (selected + 1) % N

        # reset points
        elif key in (8, 127):
            pts[:] = _init_grid_image_points(frame.shape, grid_shape, margin=0.2)

        # accept view
        elif key == 13:
            # IMPORTANT: convert to Point2f format for OpenCV: float32 and (N,2) [1](https://www.geeksforgeeks.org/python/how-to-create-requirements-txt-file-in-python/)[3](https://note.nkmk.me/en/python-pip-install-requirements/)
            imgp = pts.astype(np.float32).reshape(N, 2)
            objpoints.append(objp.copy())   # (N,3) float32
            imgpoints.append(imgp.copy())   # (N,2) float32
            view_idx += 1
            print(f"Accepted view {view_idx}/{n_views}: obj {objp.shape} {objp.dtype}, img {imgp.shape} {imgp.dtype}")
            frozen = False
            if view_idx >= n_views:
                break

        # move with WASD
        elif key == ord('w'):
            pts[selected][1] -= step
        elif key == ord('s'):
            pts[selected][1] += step
        elif key == ord('a'):
            pts[selected][0] -= step
        elif key == ord('d'):
            pts[selected][0] += step

        # move with arrow keys (extended)
        kex = cv2.waitKeyEx(1)
        if kex == 2490368:      # up
            pts[selected][1] -= step
        elif kex == 2621440:    # down
            pts[selected][1] += step
        elif kex == 2424832:    # left
            pts[selected][0] -= step
        elif kex == 2555904:    # right
            pts[selected][0] += step

        # clamp
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    cv2.destroyWindow(win)

    if len(objpoints) < 1:
        raise RuntimeError("No accepted views. Calibration aborted.")

    # Sanity checks before calling OpenCV
    for i in range(len(objpoints)):
        assert objpoints[i].dtype == np.float32 and objpoints[i].shape == (N, 3), (objpoints[i].dtype, objpoints[i].shape)
        assert imgpoints[i].dtype == np.float32 and imgpoints[i].shape == (N, 2), (imgpoints[i].dtype, imgpoints[i].shape)

    # Calibrate (expects vector-of-vectors of Point3f / Point2f) [1](https://www.geeksforgeeks.org/python/how-to-create-requirements-txt-file-in-python/)[2](https://github.com/filoucool/Python-Pick-and-Place/blob/master/detector.py)
    rms, K, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h),
        cameraMatrix=None, distCoeffs=None,
        flags=flags
    )

    print("\n=== Calibration result ===")
    print("RMS reprojection error:", rms)
    print("K:\n", K)
    print("distCoeffs:\n", distCoeffs.ravel())

    return K, distCoeffs, rms, rvecs, tvecs, (w, h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--grid", default="4x4")
    ap.add_argument("--spacing_mm", type=float, default=25.0)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--out", default="camera_calib.npz")
    args = ap.parse_args()

    rows, cols = map(int, args.grid.lower().split("x"))

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.device}")

    try:
        K, dist, rms, rvecs, tvecs, image_size = run_grid_calibration_ui(
            cap,
            grid_shape=(rows, cols),
            spacing_mm=args.spacing_mm,
            n_views=args.views
        )
        print("\n=== Calibration result ===")
        print("RMS reprojection error:", rms)
        print("K:\n", K)
        print("distCoeffs:\n", dist.ravel())
        save_calibration_npz(args.out, K, dist, image_size)
        print(f"Saved calibration to: {args.out}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()