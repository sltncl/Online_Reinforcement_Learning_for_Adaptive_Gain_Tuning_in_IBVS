from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    """
    Launch description for IBVS experiments with UR simulation.

    This launch file:
        - Includes the UR simulation + MoveIt launch file
        - Starts the cube mover node (publishes cube trajectory in Gazebo)
        - Starts the IBVS node (main control loop)
    """

    # === Include UR simulation with MoveIt ===
    ur_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('ur_simulation_gz'), 'launch', 'ur_sim_moveit.launch.py']
            )
        )
    )

    # === Cube mover node ===
    cube_node = Node(
        package='ibvs_rl',
        executable='cube_mover',
        name='cube_mover',
        output='screen',
        parameters=[{'use_sim_time': True}],  # Use simulated clock if available
    )

    # === IBVS control node ===
    ibvs_node = Node(
        package='ibvs_rl',
        executable='ibvs_node',
        name='ibvs_node',
        output='screen',
        parameters=[{'use_sim_time': True}],  # Use simulated clock if available
    )

    # === Launch description ===
    return LaunchDescription([
        ur_sim_launch,
        cube_node,
        ibvs_node
    ])