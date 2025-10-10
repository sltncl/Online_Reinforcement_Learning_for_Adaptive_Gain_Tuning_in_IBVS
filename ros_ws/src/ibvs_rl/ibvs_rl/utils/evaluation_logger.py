import os
import csv
import datetime


class EvaluationLogger:
    """
    Evaluation Logger for IBVS (Image-Based Visual Servoing) experiments.

    This class creates a timestamped CSV file and logs evaluation metrics:
        - Time (timestamp)
        - Reward (per step/episode)
        - Cumulative reward
        - Gain (action parameter selected by the agent)
        - Action computation time
        - Error vector (10 values for 5 features in image plane)

    Each call to `log()` appends one row to the CSV file.
    """

    def __init__(self, base_dir="~/ros_workspace/src/utils/eval_logs"):
        """
        Initialize the evaluation logger.

        Args:
            base_dir (str): Base directory where evaluation logs will be stored.
                            A timestamped CSV file is created inside this directory.
        """
        # Create timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir = os.path.expanduser(base_dir)
        os.makedirs(dataset_dir, exist_ok=True)
        self.path = os.path.join(dataset_dir, f"ibvs_eval_{timestamp}.csv")

        # === CSV header fields ===
        self.fields = (
            ["time", "reward", "cumulative_reward", "gain", "act_time"] +
            [f"e_{i}" for i in range(10)]
        )

        # Open CSV file and write header
        self.file = open(self.path, mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.fields)

        # Initialize cumulative reward tracker
        self.cumulative = 0.0

    def log(self, reward, gain, e, time=None, act_time=None):
        """
        Log a single evaluation step.

        Args:
            reward (float): Reward obtained at this step.
            gain (float): Action parameter (lambda) selected by the agent.
            e (array-like): Error vector (10 values for 5 features).
            time (float, optional): Timestamp. If None, current system time is used.
            act_time (float, optional): Computation time in seconds. Defaults to 0.0.
        """
        if time is None:
            time = datetime.datetime.now().timestamp()
        if act_time is None:
            act_time = 0.0

        # Update cumulative reward
        self.cumulative += reward

        # Construct row: metrics + error vector
        row = [time, float(reward), self.cumulative, float(gain), float(act_time)] \
              + [float(x) for x in e.flatten()]

        # Write row to CSV and flush buffer
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        """
        Close the CSV file safely.
        """
        if self.file:
            self.file.close()
            self.file = None