import numpy as np
import cv2


class NoiseInjector:
    """
    Noise and distortion manager to simulate real camera imperfections.

    Provides methods to:
        - Perturb intrinsic parameters (fx, fy, cx, cy)
        - Apply radial/tangential image distortion
        - Inject depth noise with Gaussian model and occasional outliers
        - Apply Gaussian blur to images

    Useful for testing robustness of vision-based algorithms under realistic sensor conditions.
    """

    def __init__(self, node, enable=True):
        """
        Initialize the noise injector.

        Args:
            node: ROS2 node (used for parameter access and logging).
            enable (bool): If False, no noise or distortion is applied.
        """
        self.node = node
        self.enable = enable

        # Standard deviations and noise model parameters
        self.fx_rel_std = 0.05   # Relative std for focal length perturbation
        self.cx_std     = 10.0   # Std deviation for principal point x
        self.cy_std     = 10.0   # Std deviation for principal point y
        self.alpha      = 0.01   # Depth noise proportionality factor
        self.beta       = 0.002  # Depth noise offset

    # -----------------------------
    # Intrinsics perturbation
    # -----------------------------
    def perturb_intrinsics(self, fx, fy, cx, cy):
        """
        Apply Gaussian noise to camera intrinsic parameters.

        Args:
            fx, fy (float): Focal lengths.
            cx, cy (float): Principal point coordinates.

        Returns:
            tuple: Noisy intrinsics (fx_noisy, fy_noisy, cx_noisy, cy_noisy).
        """
        if not self.enable:
            return fx, fy, cx, cy

        fx_noisy = fx * (1.0 + np.random.normal(0, self.fx_rel_std))
        fy_noisy = fy * (1.0 + np.random.normal(0, self.fx_rel_std))
        cx_noisy = cx + np.random.normal(0, self.cx_std)
        cy_noisy = cy + np.random.normal(0, self.cy_std)
        return fx_noisy, fy_noisy, cx_noisy, cy_noisy

    # -----------------------------
    # Image distortion
    # -----------------------------
    def distort_image(self, img, fx, fy, cx, cy):
        """
        Apply radial and tangential distortion to an image.

        Args:
            img (np.ndarray): Input image.
            fx, fy, cx, cy (float): Camera intrinsics.

        Returns:
            np.ndarray: Distorted image.
        """
        if not self.enable:
            return img

        h, w = img.shape[:2]
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float32)

        # Randomly sample distortion coefficients
        self.k1_current = np.random.uniform(-0.3, -0.1)   # Radial distortion (1st order)
        self.k2_current = np.random.uniform(0.01, 0.1)    # Radial distortion (2nd order)
        self.p1_current = np.random.uniform(-0.02, 0.02)  # Tangential distortion
        self.p2_current = np.random.uniform(-0.02, 0.02)

        dist_coeffs = np.array(
            [self.k1_current, self.k2_current, self.p1_current, self.p2_current, 0],
            dtype=np.float32
        )

        # Apply distortion mapping
        map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, K, (w, h), cv2.CV_32FC1)
        distorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        return distorted

    # -----------------------------
    # Depth noise
    # -----------------------------
    def perturb_depth(self, Z):
        """
        Apply Gaussian noise and occasional outliers to depth measurement.

        Noise model: σ = α*Z + β

        Args:
            Z (float): Depth value.

        Returns:
            float: Noisy depth value (may include outliers or NaN).
        """
        if not self.enable:
            return Z

        # Gaussian noise model
        sigma = self.alpha * Z + self.beta
        Z_noisy = Z + np.random.normal(0, sigma)

        # Random outliers (~5% probability)
        if np.random.rand() < 0.05:
            if np.random.rand() < 0.5:
                Z_noisy = Z * np.random.uniform(0.01, 0.2)   # Almost zero
            else:
                Z_noisy = Z * np.random.uniform(5.0, 10.0)   # Unrealistically far

        # Random NaNs (~2% probability)
        if np.random.rand() < 0.02:
            return np.nan

        return Z_noisy

    # -----------------------------
    # Blur injection
    # -----------------------------
    def blur_injection(self, cv_image):
        """
        Apply Gaussian blur with random kernel size.

        Args:
            cv_image (np.ndarray): Input image.

        Returns:
            np.ndarray: Blurred image.
        """
        if not self.enable:
            return cv_image

        k = np.random.choice([1, 3, 5, 7, 9, 11])  # Random odd kernel size
        return cv2.GaussianBlur(cv_image, (k, k), 0)