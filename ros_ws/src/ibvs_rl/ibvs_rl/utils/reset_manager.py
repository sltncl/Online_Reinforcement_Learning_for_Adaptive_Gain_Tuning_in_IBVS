from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_srvs.srv import SetBool


class ResetManager:
    """
    Reset Manager for robotic manipulator control.

    This class coordinates:
        - Pausing the servo controller
        - Publishing a reset trajectory to bring the manipulator
          to a predefined joint configuration
        - Resuming normal IBVS operation after reset
    """

    def __init__(self, node, joint_names):
        """
        Initialize the ResetManager.

        Args:
            node: ROS2 node used for logging and communication.
            joint_names (list[str]): List of joint names for the manipulator.
        """
        self.node = node

        # Publisher for joint trajectory commands
        self.reset_pub = node.create_publisher(
            JointTrajectory, '/servo_controller/joint_trajectory', 10
        )

        # Client for pausing/resuming the servo node
        self.pause_client = node.create_client(SetBool, '/servo_node/pause_servo')

        # Wait until the pause service is available
        while not self.pause_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warn("Waiting for /servo_node/pause_servo service...")

        self.resetting = False
        self.joint_names = joint_names

    def reset_pose(self, reset_positions, duration_sec=10):
        """
        Pause the servo controller and send a reset trajectory.

        Args:
            reset_positions (list[float]): Target joint positions for reset.
            duration_sec (int): Duration of the reset motion in seconds.
        """
        self.resetting = True

        # Pause servo node
        req = SetBool.Request()
        req.data = True
        self.pause_client.call_async(req)
        self.node.get_logger().info("Servo paused, resetting manipulator...")

        # Build trajectory message
        msg = JointTrajectory()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.joint_names = self.joint_names

        reset_point = JointTrajectoryPoint()
        reset_point.positions = reset_positions
        reset_point.velocities = [0.0] * len(self.joint_names)
        reset_point.time_from_start.sec = duration_sec

        msg.points = [reset_point]

        # Publish reset trajectory
        self.reset_pub.publish(msg)

    def end_reset(self):
        """
        Resume servo controller after reset is complete.
        """
        self.node.get_logger().info("Reset completed, resuming IBVS.")
        req = SetBool.Request()
        req.data = False
        self.pause_client.call_async(req)
        self.resetting = False