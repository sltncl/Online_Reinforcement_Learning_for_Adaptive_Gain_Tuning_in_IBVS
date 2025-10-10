from ibvs_rl.vision.apriltag_detector import AprilTagDetector
from ibvs_rl.vision.yolo_detector import YOLODetector


class VisionManager:
    """
    Vision Manager providing a unified interface for different detection methods.

    Currently supports:
        - AprilTag detection
        - YOLO-based object detection

    The manager instantiates the appropriate detector based on the chosen method
    and delegates detection calls to it.
    """

    def __init__(self, method="apriltag", **kwargs):
        """
        Initialize the VisionManager.

        Args:
            method (str): Detection method to use ("apriltag" or "yolo").
            **kwargs: Additional keyword arguments passed to the detector constructor.
        """
        if method == "apriltag":
            self.detector = AprilTagDetector(**kwargs)
        elif method == "yolo":
            self.detector = YOLODetector(**kwargs)
        else:
            raise ValueError(f"Unsupported detection method: {method}")

        self.method = method

    def detect(self, image):
        """
        Perform detection using the selected vision method.

        Args:
            image (np.ndarray): Input image (BGR format).

        Returns:
            Detection results as defined by the underlying detector:
                - AprilTagDetector: (box_points, center)
                - YOLODetector: list of detections (bounding boxes, classes, scores)
        """
        return self.detector.detect(image)