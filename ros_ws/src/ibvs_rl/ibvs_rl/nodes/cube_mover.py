import math
import subprocess
import rclpy
from rclpy.node import Node


class CubeMover(Node):
    """
    ROS 2 Node that periodically updates the position of a cube in Gazebo.
    The cube follows either a circular trajectory or a perturbed circular trajectory,
    depending on the selected scenario.
    """

    def __init__(self):
        super().__init__('cube_mover')

        # === Scenario parameter ===
        # scenario = 1 → fixed circular trajectory
        # scenario = 2 → perturbed circular trajectory (sinusoidal radius variation)
        self.declare_parameter("scenario", 1)
        self.scenario = self.get_parameter("scenario").get_parameter_value().integer_value

        # === Common trajectory parameters ===
        self.radius = 0.10          # Default radius for circular trajectory [m]
        self.center_x = -0.55       # X-coordinate of circle center [m]
        self.center_y = 0.0         # Y-coordinate of circle center [m]
        self.z = 1.04               # Fixed Z height of the cube [m]
        self.angle = 0.0            # Current angular position [rad]
        self.speed = 0.8            # Angular velocity [rad/s]
        self.timer_period = 0.2     # Timer callback period [s]

        # === Timer for periodic updates ===
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info(f"CubeMover started with scenario {self.scenario}")

    def timer_callback(self):
        """
        Periodic callback that computes the cube's new position
        and sends it to Gazebo via the set_pose service.
        """

        # === SCENARIO 1: fixed circular trajectory ===
        if self.scenario == 1:
            r = self.radius
            x = self.center_x + r * math.cos(self.angle)
            y = self.center_y + r * math.sin(self.angle)

        # === SCENARIO 2: perturbed circular trajectory ===
        elif self.scenario == 2:
            r_base = 0.098   # Base radius [m]
            amp = 0.002      # Amplitude of sinusoidal perturbation [m]

            # Oscillating radius with sinusoidal perturbation
            r = r_base + amp * math.sin(5 * self.angle)  # Frequency factor = 5

            # Clamp radius to safe bounds (ensures r ∈ [0.096, 0.10])
            r = max(0.096, min(r, 0.10))

            x = self.center_x + r * math.cos(self.angle)
            y = self.center_y + r * math.sin(self.angle)

        else:
            # Unknown scenario → log warning and skip update
            self.get_logger().warn(f"Scenario {self.scenario} not recognized.")
            return

        # === Send updated pose to Gazebo ===
        pose_str = f'name: "red_cube", position: {{x: {x:.3f}, y: {y:.3f}, z: {self.z}}}'

        try:
            subprocess.run([
                'gz', 'service', '-s', '/world/default/set_pose',
                '--reqtype', 'gz.msgs.Pose',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000',
                '--req', pose_str
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        except subprocess.CalledProcessError:
            self.get_logger().error("Failed to call set_pose service.")

        # === Increment angular position for next iteration ===
        self.angle += self.speed * self.timer_period


def main(args=None):
    """
    Entry point for the CubeMover node.
    Initializes ROS 2, starts the node, and handles shutdown.
    """
    rclpy.init(args=args)
    node = CubeMover()
    try:
        rclpy.spin(node)  # Keep node alive until interrupted
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()