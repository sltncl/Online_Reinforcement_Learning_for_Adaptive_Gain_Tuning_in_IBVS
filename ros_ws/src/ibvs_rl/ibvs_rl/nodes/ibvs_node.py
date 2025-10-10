import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import time
import cv2
import numpy as np
from geometry_msgs.msg import TwistStamped
from moveit_msgs.srv import ServoCommandType

from ibvs_rl.vision.vision_utils import build_target_square
from ibvs_rl.rl.reward import compute_reward
from ibvs_rl.rl.caql_agent import CAQLAgent
from ibvs_rl.rl.qlearning_agent import QlearningAgent
from ibvs_rl.utils.dataset_logger import DatasetLogger
from ibvs_rl.utils.reset_manager import ResetManager
from ibvs_rl.vision.vision_manager import VisionManager
from ibvs_rl.utils.noise_injector import NoiseInjector
from ibvs_rl.utils.elmpso import ELMPSO
from ibvs_rl.utils.evaluation_logger import EvaluationLogger

from controller_manager_msgs.srv import ListControllers


# ============================================================
# IBVSNode
# ============================================================
# This node implements an Image-Based Visual Servoing (IBVS) control
# loop augmented with a reinforcement learning (RL) policy to adapt
# a scalar gain (lambda). It integrates:
#   - vision: apriltag or yolo based detection of 2D features
#   - depth: per-pixel depth lookup or fallback to an ELM-PSO estimator
#   - RL agent: CAQL or Q-learning to adapt the control gain
#   - MoveIt Servo: publishes delta_twist_cmds for robot low-level control
# ============================================================
class IBVSNode(Node):
    """
    ROS2 node implementing IBVS with RL-based gain adaptation.

    Responsibilities
    ----------------
    - Subscribe to camera RGB, depth and camera_info topics.
    - Detect visual features and compute the visual error vector.
    - Build the interaction matrix (L) using either measured depth or an
      ELM-PSO fallback when depth is invalid/out of range.
    - Use the RL agent to select a scalar gain and compute a 6D camera
      velocity which is remapped and published to MoveIt Servo as
      delta_twist_cmds.
    - Optional dataset and evaluation logging for offline analysis.

    Key design notes
    ----------------
    - Depth reliability check: accepted Z in (0.25, 5.0) meters; otherwise
      fallback to the learned ELM model.
    - Vision method is configurable: "apriltag" uses tag corners + center;
      "yolo" uses bounding-box center only.
    - The node does not change robot configuration directly; it publishes
      TwistStamped messages to the servo node and relies on controllers.
    """
    def __init__(self):
        """
        Initialize the IBVS node, its dependencies and ROS interfaces.

        Side effects:
        - Creates subscribers for camera info, RGB and depth.
        - Creates publishers for MoveIt Servo commands and debug images.
        - Attempts to set MoveIt Servo command type to delta_twist_cmds.
        - Instantiates RL agent, vision manager, noise injector and loggers.
        """
        super().__init__('ibvs_node')

        # --------------------------
        # Internal state / tuning
        # --------------------------
        self.side = 40              # default side length (pixels) for desired square
        self.use_elmpso = False     # whether to use ELM-PSO fallback for L

        # --------------------------
        # ELM-PSO adaptive model
        # --------------------------
        # Used when depth is unreliable. Provides a predicted interaction
        # matrix (L) based on the observed feature error vector.
        self.elm_model = ELMPSO()

        # --------------------------
        # Controller management
        # --------------------------
        # We query controller_manager to ensure required controllers are active
        # before enabling the IBVS loop (prevents commanding robot during startup).
        self.controllers_ready = False
        self.controller_client = self.create_client(ListControllers, '/controller_manager/list_controllers')
        # Periodically poll controllers (non-blocking callback every 2 seconds)
        self.create_timer(2.0, self.check_controllers)
        
        # --------------------------
        # Image conversion utilities
        # --------------------------
        self.bridge = CvBridge()

        # --------------------------
        # Camera intrinsics & state
        # --------------------------
        # fx, fy, cx, cy are populated from CameraInfo; Z holds the last depth.
        self.fx = self.fy = self.cx = self.cy = None
        self.Z = 1.0
        self.u_star = self.v_star = None    # desired image coordinates (optical center)
        self.depth_image = None             # last received depth image as numpy array

        # --------------------------
        # Noise injection for robustness testing
        # --------------------------
        self.noise_enable = False
        self.noise = NoiseInjector(self, enable=self.noise_enable)

        # --------------------------
        # Vision pipeline
        # --------------------------
        # Configure detection method; VisionManager encapsulates detectors.
        self.method = "apriltag" # supported: "apriltag", "yolo"
        self.vision_manager = VisionManager(method=self.method)

        # --------------------------
        # Logging utilities (optional)
        # --------------------------
        self.declare_parameter("dataset_saver", False)
        self.save_dataset = self.get_parameter("dataset_saver").get_parameter_value().bool_value
        self.logger = DatasetLogger() if self.save_dataset else None

        self.declare_parameter("eval_logger", True)
        self.save_eval = self.get_parameter("eval_logger").get_parameter_value().bool_value
        self.eval_logger = EvaluationLogger() if self.save_eval else None

        # --------------------------
        # Configure MoveIt Servo command mode
        # --------------------------
        # We explicitly request delta_twist_cmds to ensure expected command semantics.
        self.command_type_client = self.create_client(ServoCommandType, '/servo_node/switch_command_type')
        while not self.command_type_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /servo_node/switch_command_type service...')
        req = ServoCommandType.Request()
        req.command_type = 1 # delta_twist_cmds
        future = self.command_type_client.call_async(req)
        # Block until the request completes to guarantee command type is set.
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('Command type set to delta_twist_cmds (1)')
        else:
            self.get_logger().error('Failed to call switch_command_type service')

        # --------------------------
        # ROS subscriptions
        # --------------------------
        # CameraInfo -> populate intrinsics once
        self.create_subscription(CameraInfo, '/wrist_mounted_camera/camera_info', self.camera_info_callback, 1)
        # RGB image -> main control loop trigger
        self.create_subscription(Image, '/wrist_mounted_camera/image', self.image_callback, 1)
        # Depth image -> used to read per-pixel depth values
        self.create_subscription(Image, '/wrist_mounted_camera/depth_image', self.depth_callback, 1)

        # --------------------------
        # ROS publishers
        # --------------------------
        # Publish delta twist commands expected by MoveIt Servo
        self.servo_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        # Optional debug image publisher (annotated visualization)
        self.debug_image_pub = self.create_publisher(Image, '/ibvs/debug_image', 1)

        # --------------------------
        # Reset manager for joint safety
        # --------------------------
        # The ResetManager handles coordinated joint resets and exposes a
        # flag (resetting) that temporarily disables IBVS processing.
        self.reset_manager = ResetManager(self, [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ])
       # Example usage (commented): self.reset_manager.reset_pose([0.0, -1.41, -1.2, -2.12, 1.67, 0.0])
       

        self.get_logger().info("IBVS node launched with MoveIt Servo")

        # --------------------------
        # Reinforcement Learning agent selection
        # --------------------------
        # The RL agent outputs a scalar gain (lambda) used to scale the
        # pseudo-inverse velocity command v_c = -lambda * L_pinv * e
        self.agent_type = "caql" # options: "qlearning" or "caql"
        if self.agent_type == "caql":
            self.agent = CAQLAgent(self, state_dim=6, action_dim=1)
        elif self.agent_type == "qlearning":
            self.agent = QlearningAgent(self)

    # ============================================================
    # Controller status check
    # ============================================================
    def check_controllers(self):
        """
        Periodically query controller_manager and enable IBVS once required controllers
        are active.

        Required controllers (example):
          - joint_state_broadcaster
          - servo_controller
          - robotiq_gripper_controller

        Rationale:
          - Avoid sending motion commands until the robot's control stack is ready.
        """
        if not self.controller_client.service_is_ready():
            self.get_logger().info("Waiting for /controller_manager/list_controllers service...")
            return
        
        req = ListControllers.Request()
        future = self.controller_client.call_async(req)

        def callback(fut):
            try:
                result = fut.result()
                active = {c.name: c.state for c in result.controller}
                needed = ["joint_state_broadcaster", "servo_controller", "robotiq_gripper_controller"]

                # Enable node only when all needed controllers are active
                if all(active.get(ctrl) == "active" for ctrl in needed):
                    if not self.controllers_ready:
                        self.controllers_ready = True
                        self.get_logger().info("All controllers active, IBVS enabled!")
            except Exception as e:
                # Log and continue; controller failures should not crash node.
                self.get_logger().error(f"Error checking controllers: {e}")
        # Use done-callback to avoid blocking the ROS spinning thread.
        future.add_done_callback(callback)

    # ============================================================
    # Camera intrinsics callback
    # ============================================================
    def camera_info_callback(self, msg: CameraInfo):
        """
        Receive CameraInfo once and extract intrinsics.

        Effects:
          - Sets fx, fy, cx, cy on first callback (idempotent).
          - Optionally perturbs intrinsics via NoiseInjector for robustness tests.
          - Sets u_star, v_star as desired optical center for visual servoing.
        """
        if self.fx is None:
            # CameraInfo.k is row-major 3x3 intrinsic matrix [k0..k8]
            self.fx = msg.k[0]
            self.fy = msg.k[4] 
            self.cx = msg.k[2] 
            self.cy = msg.k[5]

             # Apply synthetic perturbations if noise injection enabled
            self.fx, self.fy, self.cx, self.cy = self.noise.perturb_intrinsics(self.fx, self.fy, self.cx, self.cy)

            # Desired image point (optical center used as setpoint)
            self.u_star = self.cx
            self.v_star = self.cy

    # ============================================================
    # Depth image callback
    # ============================================================
    def depth_callback(self, msg: Image):
        """
        Convert incoming depth Image message to a NumPy array.

        Notes:
        - Expects depth in a format compatible with CvBridge passthrough encoding.
        - On conversion error, depth_image is set to None to force fallback behavior.
        """
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            # Use warn instead of error: depth may be temporarily unavailable.
            self.get_logger().warn(f"Depth image conversion error: {e}")
            self.depth_image = None

    # ============================================================
    # Main image callback — IBVS + RL control loop
    # ============================================================
    def image_callback(self, msg: Image):
        """
        Primary image processing and control routine.

        Workflow:
          1. Safety checks (reset in progress, controllers ready, intrinsics/depth present).
          2. Convert RGB image and apply optional synthetic noise.
          3. Detect features (apriltag / yolo).
          4. Build observed and desired feature sets.
          5. For each feature:
             - Attempt depth lookup; if invalid, mark for ELM-PSO fallback.
             - Compute normalized image-space error and per-feature interaction matrix L_i.
          6. Assemble global error vector (e) and full interaction matrix (L).
             Use pseudo-inverse or learned ELM model as appropriate.
          7. Form state vector for RL agent: state = L_pinv @ e.
          8. Query RL agent for scalar gain (lambda) and form camera velocity v_c.
          9. Compute reward and update RL agent.
         10. Publish TwistStamped to MoveIt Servo and debug image.

        Failure modes:
          - Missing intrinsics or depth -> loop returns early.
          - Depth out-of-range or invalid -> fallback to ELM-PSO for interaction matrix.
          - Invalid computed velocity dimension -> skip publish and log warning.
        """
        # --------------------------
        # Safety and readiness checks
        # --------------------------
        # Avoid commanding robot while reset is active.
        if self.reset_manager.resetting:
            self.get_logger().debug("IBVS disabled during reset")
            return 
        # Ensure controllers are active to avoid lost commands.
        if not self.controllers_ready:
            self.get_logger().debug("Controllers not yet active, skipping IBVS...")
            return
        # Ensure camera intrinsics are available.
        if any(val is None for val in [self.fx, self.fy, self.cx, self.cy]):
            self.get_logger().warn("Waiting for camera intrinsics...")
            return
        # Ensure depth image present (depth is required for interaction matrix).
        if self.depth_image is None:
            self.get_logger().warn("Waiting for depth image...")
            return

        # --------------------------
        # Image conversion and synthetic noise
        # --------------------------
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Apply optional blur/distortion to simulate sensor noise
            cv_image = self.noise.blur_injection(cv_image)
            cv_image = self.noise.distort_image(cv_image, self.fx, self.fy, self.cx, self.cy)
        except Exception as e:
            # Conversion failure; do not proceed with potentially invalid image.
            self.get_logger().warn(f"RGB image conversion error: {e}")
            return

        # --------------------------
        # Feature detection
        # --------------------------
        corners, center = self.vision_manager.detect(cv_image)
        if corners is None or center is None:
            # No target detected; safe to skip control iteration.
            self.get_logger().info("Target not found")
            return
        
        # --------------------------
        # Desired feature configuration
        # --------------------------
        # For apriltag: use tag corners + center; for yolo: use bounding-box center.
        if self.method == "apriltag":
            target_corners = build_target_square(self.u_star, self.v_star, side=self.side)
            obs_points = [tuple(c) for c in corners] + [center]
            des_points = [tuple(c) for c in target_corners] + [(self.u_star, self.v_star)]
        elif self.method == "yolo":
            obs_points = [center]
            des_points = [(self.u_star, self.v_star)]
        else:
            # Defensive: if method is unknown, skip processing.
            self.get_logger().error(f"Unsupported vision method: {self.method}")
            return

        # --------------------------
        # Visual error and interaction matrix computation
        # --------------------------
        e_list = []                 # flattened error list for all features (2n elements)
        L_list = []                 # list of per-feature 2x6 interaction matrices
        self.use_elmpso = False     # reset fallback flag for this frame

        for (u, v), (u_d, v_d) in zip(obs_points, des_points):
            # Attempt to read depth at integer pixel coordinates
            try:
                # Ensure indices are within bounds before indexing depth image
                if 0 <= v < self.depth_image.shape[0] and 0 <= u < self.depth_image.shape[1]:
                    Z_val = float(self.depth_image[int(v), int(u)])
                    # Allow NoiseInjector to perturb depth (for robustness experiments)
                    self.Z = self.noise.perturb_depth(Z_val)
                    # Validate depth range and NaN
                    if not(self.Z > 0.25 and self.Z < 5.0 and not np.isnan(self.Z)):
                        # Depth invalid or outside operational range -> fallback
                        self.use_elmpso = True
                        self.get_logger().info("Uso ELM!")
                else:
                    # Coordinates out of bounds -> fallback (log details)
                    self.get_logger().info("Uso ELM!")
                    self.use_elmpso = True
                    self.get_logger().warn("Feature coordinates out of depth image bounds")
            except Exception as e:
                # Any error while accessing depth triggers fallback to learned model
                self.get_logger().info("Uso ELM!")
                self.use_elmpso = True
                self.get_logger().warn(f"Depth access error: {e}")

            # Normalized pixel error (image-space), scaled by focal lengths
            e_list.extend([(u - u_d) / self.fx, (v - v_d) / self.fy])

            if not self.use_elmpso:
                # Compute interaction matrix (L_i) for the current feature
                # Using the standard IBVS 2x6 interaction matrix for a point feature:

                # Image coordinates in normalized camera frame
                x = (u - self.cx) / self.fx
                y = (v - self.cy) / self.fy
                # Interaction matrix (2x6) for each feature
                L_i = np.array([
                    [-1/self.Z,     0,   x/self.Z,     x*y, -(1 + x**2),     y],
                    [      0, -1/self.Z,   y/self.Z, 1 + y**2,       -x*y,    -x]
                ])
                L_list.append(L_i)

            # Draw debug markers for observed and desired points and their error vector
            cv2.circle(cv_image, (int(u), int(v)), 5, (255, 0, 0), 2)           # observed
            cv2.circle(cv_image, (int(u_d), int(v_d)), 5, (255, 0, 0), 2)       # desired
            cv2.line(cv_image, (int(u), int(v)), (int(u_d), int(v_d)), (255, 255, 0), 1)

        # Construct global error vector (2n x 1)
        e = np.array(e_list, dtype=float).reshape(-1, 1)  
        # Choose how to obtain L and its pseudo-inverse:
        # - If depth is valid for all features, stack per-feature L_i and compute Moore-Penrose pseudo-inverse.
        # - Otherwise, let the learned ELM-PSO model predict L and compute its pseudo-inverse.
        if not self.use_elmpso:
            # Stack per-feature interaction matrices into a (2n x 6) matrix
            L = np.vstack(L_list) 
            # Moore-Penrose pseudo-inverse for stable least-squares solution                            
            L_pinv = np.linalg.pinv(L)
        else: 
            # Predict L from ELM model given the flattened error vector
            L = self.elm_model.predict_L(e.flatten())         # expected shape: (2n x 6)
            L_pinv = self.elm_model.pseudo_inverse(L)         # expected shape: (6 x 2n)

        # --------------------------
        # RL state and action selection
        # --------------------------
        # The RL state is formed as L_pinv @ e which corresponds to the estimated
        # camera velocity for unit gain; the RL agent returns a scalar lambda
        # to scale that velocity.
        state_vec = (L_pinv @ e).reshape(-1)

        # Time the agent's decision for performance logging
        start_time = time.perf_counter()

        if self.agent_type == "caql":
            # CAQL returns (lambda, noise); warmstart uses last_action for smoother updates
            action_lambda, noise = self.agent.act(state_vec, warmstart=self.agent.last_action)
        elif self.agent_type == "qlearning":
            action_lambda = self.agent.act(state_vec)
            noise = None
        else:
            # Defensive: unknown agent type
            self.get_logger().error(f"Unsupported agent type: {self.agent_type}")
            return

        elapsed_time = time.perf_counter() - start_time   # action selection latency (s)

        # --------------------------
        # Control computation
        # --------------------------
        # Compute 6D camera velocity using scaled pseudo-inverse: v_c = -lambda * L_pinv * e
        v_c = -action_lambda * (L_pinv @ e) # shape (6 x 1)
        v_c = v_c.reshape(-1)
        # Validate computed command size prior to publishing
        if v_c.shape[0] != 6:
            self.get_logger().warn("Invalid velocity vector, skipping publish")
            return

        # --------------------------
        # Reward computation and RL update
        # --------------------------
        # e_dot is the instantaneous error derivative approximated by L @ v_c
        e_dot = (L @ v_c.reshape(-1,1)).reshape(-1,1)
        reward = compute_reward(e, e_dot, self.fx, self.fy)

        # Update agent memory / Q-table depending on agent algorithm
        if self.agent_type == "caql":
            # CAQL expects experience for replay or on-policy updates
            self.agent.remember(reward, state_vec, done=False)
        elif self.agent_type == "qlearning":
            # Q-learning update rule uses current reward and state
            self.agent.update_Q(reward,state_vec)

        # --------------------------
        # Optional logging for datasets and evaluation
        # --------------------------
        if self.logger:
            # Persist raw data for offline training/analysis
            self.logger.write_row(e, L)
        if self.eval_logger:
            # Log reward/action/time for performance evaluation
            self.eval_logger.log(reward, action_lambda, e, act_time=elapsed_time)

        # --------------------------
        # Map camera velocity to robot frame and publish
        # --------------------------
        # The mapping preserves the original IBVS sign conventions and the robot's axis mapping.
        cmd_msg = TwistStamped()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.header.frame_id = "wrist_mounted_camera_color_frame"

        # Map a 6D camera velocity (vx, vy, vz, wx, wy, wz) to Twist fields
        # Note: ordering and sign flips reflect the robot's coordinate expectations.
        cmd_msg.twist.linear.x  = float(v_c[2])
        cmd_msg.twist.linear.y  = float(-v_c[0])
        cmd_msg.twist.linear.z  = float(-v_c[1])
        cmd_msg.twist.angular.x = float(v_c[5])
        cmd_msg.twist.angular.y = float(-v_c[3])
        cmd_msg.twist.angular.z = float(-v_c[4])

        # Publish the delta twist command to MoveIt Servo
        self.servo_pub.publish(cmd_msg)
        # Log chosen gain and any exploration noise for debugging and reproducibility
        self.get_logger().info(f"Gain: {action_lambda}, Noise: {noise}")

        # --------------------------
        # Publish annotated debug image
        # --------------------------
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            # Non-fatal: debug image failure should not impact control loop
            self.get_logger().warn(f"Failed to publish debug image: {e}")

    # ============================================================
    # Node shutdown
    # ============================================================
    def destroy_node(self):
        """
        Perform graceful shutdown:
          - flush and close loggers
          - save RL agent state (checkpoint)
          - call base class destroy to cleanup ROS resources
        """
        if self.logger:
            try:
                self.logger.close()
                self.get_logger().info("Dataset saved.")
            except Exception as e:
                self.get_logger().warn(f"Error closing dataset logger: {e}")
        if self.eval_logger:
            try:
                self.eval_logger.close()
            except Exception as e:
                self.get_logger().warn(f"Error closing evaluation logger: {e}")
         # Persist agent parameters/weights for later reuse or evaluation
        try:
            self.agent.save()
        except Exception as e:
            self.get_logger().warn(f"Error saving agent state: {e}")
        # Call superclass cleanup to release ROS entities
        super().destroy_node()