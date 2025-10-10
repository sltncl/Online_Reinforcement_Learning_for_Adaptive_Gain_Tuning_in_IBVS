import os
import csv
import datetime


class DatasetLogger:
    """
    Dataset Logger for IBVS (Image-Based Visual Servoing) experiments.

    This class creates a timestamped CSV file and logs:
        - Feature errors (e): 10 values corresponding to 5 image points (u,v)
        - Interaction matrix (L): 10x6 matrix flattened into row format

    Each row in the dataset corresponds to one sample of (e, L).
    """

    def __init__(self, base_dir="~/ros_workspace/src/utils/dataset_for_elmpso"):
        """
        Initialize the dataset logger.

        Args:
            base_dir (str): Base directory where the dataset will be stored.
                            A timestamped CSV file is created inside this directory.
        """
        # Create timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir = os.path.expanduser(base_dir)
        os.makedirs(dataset_dir, exist_ok=True)
        self.dataset_path = os.path.join(dataset_dir, f'ibvs_elm_dataset_{timestamp}.csv')

        # === CSV header fields ===
        # - 10 error terms (for 5 features → 2D coordinates each)
        # - 10x6 interaction matrix entries
        self.data_fields = (
            [f"e_{i}" for i in range(10)] +
            [f"L_{i}_{j}" for i in range(10) for j in range(6)]
        )

        # Open CSV file and write header
        self.file = open(self.dataset_path, mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.data_fields)

    def write_row(self, e, L):
        """
        Write a single row of data to the CSV file.

        Args:
            e (np.ndarray): Error vector of shape (10,), containing
                            [e_u1, e_v1, ..., e_u5, e_v5].
            L (np.ndarray): Interaction matrix of shape (10, 6).
        """
        row = list(float(x) for x in e.flatten()) + list(L.flatten())
        self.writer.writerow(row)

    def close(self):
        """
        Close the CSV file safely.
        """
        if self.file:
            self.file.close()