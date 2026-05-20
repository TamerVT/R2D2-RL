import sys
import os
import cv2
import numpy as np

# Ensure your local workspace folder is on the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the environment factory components from your project
from env.env_factory import make_so101_sim

class ImagePreprocessor:
    def __init__(self, target_size=(320, 320), edge_low=50, edge_high=150):
        """
        Squeezes, grayscales, and extracts features from raw simulation arrays.
        """
        self.target_size = target_size
        self.edge_low = edge_low
        self.edge_high = edge_high

    def process(self, img_rgba, apply_augmentation=True, seed=None):
        """
        Processes a raw MuJoCo image frame into a normalized 320x320 edge map 
        with optional adversarial noise injection for CNN robust training.
        
        Input: img_rgba (H, W, 3) or (H, W, 4) numpy array
        Output: (320, 320) augmented edge map (values 0 or 255)
        """
        if img_rgba is None or img_rgba.size == 0:
            raise ValueError("Empty image array encountered during preprocessing.")

        # 1. Resize down to target dimensions
        resized = cv2.resize(img_rgba, self.target_size, interpolation=cv2.INTER_AREA)

        # 2. Convert to Grayscale safely
        if resized.shape[2] == 4:
            gray = cv2.cvtColor(resized, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

        # 3. Extract the clean base edge map using Canny
        edges = cv2.Canny(gray, self.edge_low, self.edge_high)

        # Early exit if we don't want noise (e.g., during validation/testing)
        if not apply_augmentation:
            return edges

        # Initialize local random number generator for clean seed isolation
        rng = np.random.default_rng(seed)

        # ==========================================================
        # NOISE INJECTION STAGE 1: Random Stray Line Artifacts
        # Simulates shadows, table edges, and background workspace clutter
        # ==========================================================
        num_spurious_lines = rng.integers(2, 6) # Inject between 2 and 5 stray lines
        for _ in range(num_spurious_lines):
            pt1 = tuple(rng.integers(0, 320, size=2).tolist())
            pt2 = tuple(rng.integers(0, 320, size=2).tolist())
            # Draw random edge lines directly into the map
            cv2.line(edges, pt1, pt2, color=255, thickness=rng.integers(1, 2))

        # ==========================================================
        # NOISE INJECTION STAGE 2: Salt & Pepper Pixel Distortion
        # Simulates sensor camera static or broken texture points
        # ==========================================================
        # Salt (random white pixels floating around)
        salt_density = rng.uniform(0.005, 0.02) # up to 2% pixel corruption
        num_salt = int(salt_density * edges.size)
        coords_y = rng.integers(0, edges.shape[0], size=num_salt)
        coords_x = rng.integers(0, edges.shape[1], size=num_salt)
        edges[coords_y, coords_x] = 255

        # Pepper (random black pixels biting chunks out of the true edges)
        pepper_density = rng.uniform(0.01, 0.04) # up to 4% edge fragmentation
        num_pepper = int(pepper_density * edges.size)
        coords_y = rng.integers(0, edges.shape[0], size=num_pepper)
        coords_x = rng.integers(0, edges.shape[1], size=num_pepper)
        edges[coords_y, coords_x] = 0

        # ==========================================================
        # NOISE INJECTION STAGE 3: Structural Morphological Jitter
        # Simulates camera motion blur, defocusing, and lens distortions
        # ==========================================================
        roll = rng.random()
        if roll < 0.25:
            # Dilation: Makes edge lines thicker (simulates close-up focus bloom)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            edges = cv2.dilate(edges, kernel, iterations=1)
        elif roll < 0.50:
            # Gaussian Blur + Re-thresholding: Blurs and fragments continuous lines
            edges = cv2.GaussianBlur(edges, (3, 3), 0)
            _, edges = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

        return edges

def main():
    print("Initializing SO101 simulator bundle...")
    # Initialize the simulation setup headlessly for rapid data processing
    # Adjust arguments here to match your exact make_so101_sim footprint
    bundle = make_so101_sim(with_cameras=True, headless=True, debug_print=False)
    
    # Target your specific active camera
    # Common RCS designations: 'wrist_camera', 'robotwrist', or 'front_view'
    camera_name = "robotwrist" 
    
    # Fall back to the first available camera index if the requested string isn't registered
    if hasattr(bundle, 'camera_names') and bundle.camera_names:
        if camera_name not in bundle.camera_names:
            camera_name = bundle.camera_names[0]
    print(f"Targeting rendering camera: '{camera_name}'")

    # Access the rendering architecture safely depending on your frame setup
    sim = bundle.sim
    
    # Structural check to handle diverse RCS wrapper interfaces safely
    if hasattr(bundle, 'renderer') and bundle.renderer is not None:
        renderer = bundle.renderer
    else:
        import mujoco
        renderer = mujoco.Renderer(sim.model, height=480, width=640)

    # 1. Capture a raw frame from the simulation state
    print("Capturing raw frame from scene...")
    renderer.update_scene(sim.data, camera=camera_name)
    raw_frame = renderer.render()

    # 2. Instantiate preprocessor and process the frame
    print("Processing frame through 320x320 edge pipeline...")
    preprocessor = ImagePreprocessor(target_size=(320, 320), edge_low=50, edge_high=150)
    processed_image = preprocessor.process(raw_frame)

    # 3. Validate and verify structural export array dimensions
    print(f"Pipeline executed successfully!")
    print(f"-> Raw input matrix shape:       {raw_frame.shape}")
    print(f"-> Preprocessed output shape:     {processed_image.shape}")
    
    # Save a verification copy to local storage to double-check edge quality
    output_filename = "data_collection_sample.png"
    cv2.imwrite(output_filename, processed_image)
    print(f"Diagnostic sample written to: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    main()