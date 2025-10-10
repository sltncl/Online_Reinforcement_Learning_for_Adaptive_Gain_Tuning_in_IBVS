import rclpy
from ibvs_rl.nodes.ibvs_node import IBVSNode


def main(args=None):
    """
    Entry point for the IBVSNode.

    Initializes ROS 2, creates the IBVS node, loads the agent,
    and spins the node until interrupted. On shutdown, the agent
    state is saved and resources are released cleanly.
    """
    # Initialize ROS 2 client library
    rclpy.init(args=args)

    # Create IBVS node instance
    node = IBVSNode()

    # Load agent state (e.g., model parameters, replay buffer)
    node.agent.load()

    try:
        # Keep node alive and responsive to callbacks
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        node.get_logger().info("Node shutdown requested (Ctrl+C)")
    finally:
        # Save agent state before exit
        node.agent.save()

        # Destroy node and shutdown ROS 2
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()