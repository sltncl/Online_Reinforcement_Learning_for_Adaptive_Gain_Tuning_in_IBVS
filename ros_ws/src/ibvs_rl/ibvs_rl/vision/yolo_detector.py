import numpy as np
from ultralytics import YOLO
from ibvs_rl.vision.vision_utils import order_box_points


class YOLODetector:
    """
    YOLO-based object detector wrapper.

    Loads a YOLO11 model and detects objects of a specified target class.
    Returns both the ordered bounding box corners and the object center.
    """

    def __init__(self, model_name="yolo11n.pt", target_class="sports ball"):
        """
        Initialize the YOLO detector.

        Args:
            model_name (str): YOLO model file (e.g., 'yolo11n.pt', 'yolo11s.pt').
                              Must be compatible with the Ultralytics YOLO API.
            target_class (str): Name of the class to detect (must exist in YOLO's dataset).
        """
        self.model = YOLO(model_name)
        self.target_class = target_class

    def detect(self, image):
        """
        Detect the target object in the input image.

        Args:
            image (np.ndarray): Input image (BGR format).

        Returns:
            tuple:
                - corners (list[list[float]]): Ordered list of 4 corner points (x, y).
                - center (tuple[int, int]): Pixel coordinates of the object center (u, v).
                If the target class is not detected, returns (None, None).
        """
        # Run YOLO inference
        results = self.model(image)

        # Check if detections exist
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, None

        # Iterate over detected bounding boxes
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            cls_name = self.model.names[cls_id]

            # Filter by target class
            if cls_name == self.target_class:
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = xyxy

                # Define corners: top-left, top-right, bottom-right, bottom-left
                corners = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ], dtype="float32")

                # Ensure consistent ordering of corners
                corners = order_box_points(corners)

                # Compute center of bounding box
                u_c = int((x1 + x2) / 2)
                v_c = int((y1 + y2) / 2)

                return corners.tolist(), (u_c, v_c)

        # If target class not found
        return None, None