
import cv2
import time
from ultralytics import YOLO
import numpy as np

# -----------------------------
# Preprocess
# -----------------------------
def preprocess(frame):
    size=(320, 320)
    # Prefer INTER_AREA for shrinking images
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return small

# -----------------------------
# Detector wrapper (optional)
# -----------------------------
class Net:
    def __init__(self, weights, imgsz=320, conf=0.25):
        self.model = YOLO(weights)  # load once
        self.imgsz = imgsz
        self.conf = conf

    def forward(self, img_bgr):
        # Ultralytics returns a list of Results objects (or generator with stream=True)
        # Each Results has .boxes for bbox outputs
        results = self.model(img_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)
        return results[0]  # first (and only) frame result

# -----------------------------
# Helper functions
# -----------------------------

def quat_xyzw_to_R(q):
    """Convert quaternion (x,y,z,w) to rotation matrix (3x3)."""
    x, y, z, w = q
    # Normalize to be safe
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),       2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),   1 - 2*(xx + yy)]
    ], dtype=np.float64)


def scale_intrinsics(K, sx, sy):
    """Scale intrinsic matrix when resizing image by sx, sy. [2](https://hlfshell.ai/posts/ppo-pick-and-place/)"""
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= sx  # fx
    K2[1, 1] *= sy  # fy
    K2[0, 2] *= sx  # cx
    K2[1, 2] *= sy  # cy
    return K2


def undistort_to_normalized_ray(u, v, K, dist):
    """
    Convert pixel (u,v) to undistorted normalized camera coordinates (x, y).
    If P is omitted/identity, undistortPoints returns normalized coordinates. [3](https://www.libhunt.com/l/python/topic/pick-and-place)[4](https://pypi.org/)
    """
    pts = np.array([[[u, v]]], dtype=np.float64)  # (1,1,2)
    xy = cv2.undistortPoints(pts, K, dist)        # (1,1,2) normalized if P omitted [3](https://www.libhunt.com/l/python/topic/pick-and-place)[4](https://pypi.org/)
    x, y = xy[0, 0, 0], xy[0, 0, 1]
    # Ray direction in camera frame (z=1)
    d_cam = np.array([x, y, 1.0], dtype=np.float64)
    d_cam /= np.linalg.norm(d_cam)
    return d_cam, (x, y)


def intersect_ray_with_z_plane(ray_origin, ray_dir, z_plane):
    """
    Intersect ray P(t)=O+t*D with plane z=z_plane. Returns None if parallel or behind camera.
    """
    dz = ray_dir[2]
    if abs(dz) < 1e-9:
        return None
    t = (z_plane - ray_origin[2]) / dz
    if t <= 0:
        return None
    return ray_origin + t * ray_dir

# Model (use small model for speed)

WEIGHTS = "OUR_stuff/Models_public/yolov8n.pt"
net = Net(weights=WEIGHTS, imgsz=320, conf=0.25)
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError("Could not open /dev/video0")

# Windows
cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("preprocessed", cv2.WINDOW_NORMAL)
cv2.namedWindow("detections", cv2.WINDOW_NORMAL)

# cup class
CUP_ID = 41

last = time.time()
t_start = time.time()

# ---- Camera calibration (for the ORIGINAL frame resolution, e.g. 640x480) ----
# K_full: 3x3 camera intrinsic matrix at the full frame resolution
# distCoeffs: distortion coefficients from calibration (k1,k2,p1,p2,k3[,k4,k5,k6])
K_full = np.array([
    [600.0,   0.0, 320.0],
    [  0.0, 600.0, 240.0],
    [  0.0,   0.0,   1.0]

], dtype=np.float64)

distCoeffs = np.array([-0.25, 0.08, 0.0, 0.0, -0.02], dtype=np.float64)  # example length=5

# ---- Fixed camera pose in WORLD coordinates (mm + quaternion) ----
# cam_pos_world_mm: (3,)
# cam_quat_xyzw: (4,) quaternion (x,y,z,w)
cam_pos_world_mm = np.array([0, -90, 530], dtype=np.float64)
cam_quat_xyzw    = np.array([0.92387953, 0.0, 0.0, -0.38268343], dtype=np.float64)

# ---- World plane definition (table plane) ----
# simplest: plane is Z = table_z_mm in world frame
table_z_mm = 0.0


try:
    # this is now a loop, but will later be replaced by a loop until found loop.
    # will be inside function which just returns the coordinates of the cube
    while time.time() < t_start + 60:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # FPS overlay
        now = time.time()
        fps = 1.0 / max(now - last, 1e-6)
        last = now
        cv2.putText(frame, f"{fps:5.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Preprocess
        prepd = preprocess(frame)

        # Inference on small frame
        result = net.forward(prepd)  # Results object [2](https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3)[5](https://iq.opengenus.org/different-interpolation-methods-in-opencv/)
        r = result
        boxes = r.boxes
        
        # filter out everything except cup
        mask_cup = (boxes.cls == CUP_ID)
        cup_boxes = boxes[mask_cup] if mask_cup.any() else None
        
        
        
            
        # Visualize detections
        det_small = prepd.copy()
        det_full = frame.copy()

        # Scale factors to map 128x128 -> original
        orig_h, orig_w = frame.shape[:2]
        sx = orig_w / 320.0
        sy = orig_h / 320.0

        # Draw boxes if any
        if cup_boxes is not None:  # bbox outputs live here [2](https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3)[5](https://iq.opengenus.org/different-interpolation-methods-in-opencv/)
            for b in cup_boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                cls = int(b.cls[0].cpu().numpy())
                conf = float(b.conf[0].cpu().numpy())

                name = net.model.names.get(cls, str(cls))
                label = f"{name} {conf:.2f}"
                
                # calculate angles wrt. center of camera. 2 are required. -----
                
                # center of box
                uc = 0.5 * (x1 + x2)
                vc = 0.5 * (y1 + y2)

                # correct for lens distortion
                prepd_h, prepd_w = prepd.shape[:2]
                K_prepd = scale_intrinsics(K_full, prepd_w / orig_w, prepd_h / orig_h)
                
                ray_cam, (xn, yn) = undistort_to_normalized_ray(uc, vc, K_prepd, distCoeffs) 
                
                # now get angles
                yaw   = np.arctan(xn)   # left/right
                pitch = np.arctan(yn)   # up/down
                
                # calculate the coordinates in mm using the camera's positional info ----
                
                R_wc = quat_xyzw_to_R(cam_quat_xyzw)   # rotation from camera -> world (assumed)
                ray_world = R_wc @ ray_cam
                ray_world /= np.linalg.norm(ray_world)
                
                origin_world = cam_pos_world_mm
                
                # find intersection
                P_world = intersect_ray_with_z_plane(origin_world, ray_world, table_z_mm)
                


                # Draw on small
                cv2.rectangle(det_small, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(det_small, label, (int(x1), max(0, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Map to full frame and draw
                X1, Y1, X2, Y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
                cv2.rectangle(det_full, (X1, Y1), (X2, Y2), (0, 255, 0), 2)
                cv2.putText(det_full, label, (X1, max(0, Y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # draw and map onto full frame (not needed except for debugging)
                if P_world is not None:
                    Xmm, Ymm, Zmm = P_world.tolist()
                    # optionally: show as text for debugging
                    cv2.putText(det_full, f"X={Xmm:.1f}mm Y={Ymm:.1f}mm",
                                (X1, min(orig_h - 10, Y2 + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    # no valid intersection (ray parallel to plane or pointing away)
                    Xmm = Ymm = Zmm = None

        # Show
        cv2.imshow("camera", frame)
        cv2.imshow("preprocessed", prepd)
        cv2.imshow("detections", det_full)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()


