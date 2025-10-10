import cv2
import numpy as np
import apriltag
from ibvs_rl.vision.vision_utils import order_box_points


class AprilTagDetector:
    """
    AprilTag detector wrapper.

    Provides functionality to detect AprilTags in an image and
    return both the ordered bounding box points and the tag center.
    """

    def __init__(self, tag_family="tag36h11"):
        """
        Initialize the AprilTag detector.

        Args:
            tag_family (str): AprilTag family to detect (default: "tag36h11").
        """
        self.detector = apriltag.Detector(
            apriltag.DetectorOptions(families=tag_family)
        )

    def detect(self, image, tag_id=0):
        """
        Detect a specific AprilTag in the input image.

        Args:
            image (np.ndarray): Input image (BGR format).
            tag_id (int): ID of the AprilTag to detect (default: 0).

        Returns:
            tuple:
                - box (list[list[float]]): Ordered list of 4 corner points (x, y).
                - center (tuple[int, int]): Pixel coordinates of the tag center (u, v).
                If the tag is not found, returns (None, None).
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Run AprilTag detection
        results = self.detector.detect(gray)

        # Search for the requested tag ID
        for r in results:
            if r.tag_id == tag_id:
                # Extract and order corner points
                c = np.array(r.corners, dtype="float32")
                box = order_box_points(c)

                # Compute tag center (mean of corner coordinates)
                u_c = int(np.mean(c[:, 0]))
                v_c = int(np.mean(c[:, 1]))

                return box.tolist(), (u_c, v_c)

        # If tag not found
        return None, None