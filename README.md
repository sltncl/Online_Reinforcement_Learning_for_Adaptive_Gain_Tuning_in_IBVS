# Online Reinforcement Learning for Adaptive Gain Tuning in Image-Based Visual Servoing 

This repository contains the implementation and simulation environment developed for the Master’s Thesis *“Online Reinforcement Learning for Adaptive Gain Tuning in Image-Based Visual Servoing”* at Politecnico di Bari, MSc in Automation Engineering (A.Y. 2024–2025).

The project integrates Reinforcement Learning techniques with a Visual Servoing control framework in ROS 2, enabling adaptive gain tuning for a 6-DOF robotic manipulator equipped with a Robotiq 2F-85 gripper and an Intel RealSense D415 camera.

---

## 📁 Repository Structure

```
.
├── docker_ws/                   # Docker workspace for building the development container
├── ros_ws/                      # Main ROS 2 workspace containing all custom and third-party packages
├── chown_me.sh                  # Script to fix ownership of files created as root inside the container
├── run.sh                       # Script to run the Docker container with correct volumes and permissions
├── exec.sh                      # Script to open a shell into a running container
```

---

## 🧩 ROS Workspace Overview (`ros_ws/`)

```
ros_ws/
 └── src/
      ├── ibvs_rl/               # Custom ROS 2 package developed for the thesis
      │    ├── ibvs_rl/          # Main source folder implementing the IBVS-RL controller
      │    │    ├── main.py      # Entry point for the adaptive IBVS controller
      │    │    ├── nodes/       # ROS nodes handling communication and control
      │    │    ├── rl/          # Reinforcement learning algorithms (QL, CAQL)
      │    │    ├── utils/       # Support functions and helper modules
      │    │    └── vision/      # Visual feature extraction (AprilTag, YOLO)
      │    ├── launch/           # ROS 2 launch files (e.g., main.launch.py)
      │    ├── setup.py          # Package setup file
      │    └── package.xml       # ROS 2 package metadata
      ├── ur_moveit_config/      # Official MoveIt configuration package for UR robots
      ├── ur_simulation_gz/      # Gazebo simulation package for UR robots
      ├── robotiq_description/   # Robotiq gripper description package
      ├── utils/                 # Folder for trained ELM-PSO and RL networks, and log datasets
      └── picknik_accessories/   # Accessory packages for MoveIt (camera and adapter support)
```

---

## 🔗 Third-Party Packages

The following ROS 2 packages were cloned and adapted for the project’s simulation environment:

- **Robotiq Gripper Description:**  
  [PickNik Robotics – ros2_robotiq_gripper](https://github.com/PickNikRobotics/ros2_robotiq_gripper/tree/main/robotiq_description)

- **Camera and Adapter (Intel RealSense D415):**  
  [PickNik Robotics – picknik_accessories](https://github.com/PickNikRobotics/picknik_accessories)

- **Universal Robots Gazebo Simulation:**  
  [Universal Robots – ROS2 Gazebo Simulation](https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation)

- **Universal Robots MoveIt Configuration:**  
  [Universal Robots – ROS2 Driver (MoveIt Config)](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/tree/humble/ur_moveit_config)

All these repositories were **modified and integrated** to simulate a **UR5e manipulator** equipped with a **Robotiq 2F-85 gripper** and an **Intel RealSense D415 camera** within a unified Docker-based ROS 2 environment.

---

## ⚙️ Build and Execution Instructions

The project is designed to run inside a Docker container for full platform compatibility across macOS, Linux, and Windows.

### 1. Build the Docker Container
```bash
cd docker_ws
chmod +x build.sh
./build.sh
```

### 2. Run the Container
From the root of the repository:
```bash
chmod +x run.sh exec.sh chown_me.sh
./run.sh
```

### 3. Connect to the Development Environment
Open your browser and navigate to:
```
http://localhost:6080
```
Then open a terminal within the VNC interface to access the development environment (Visual Studio terminal).

### 4. Access the Container and Launch the Simulation
Open a **new terminal** on your host machine and connect to the running container:
```bash
./exec.sh
```

Then, **inside the container terminal**, run:
```bash
source /opt/ros/jazzy/setup.bash
export GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH}:/root/ros_workspace/src/
cd /root/ros_workspace
colcon build
source install/setup.bash
ros2 launch ibvs_rl main.launch.py
```

This will start the full IBVS-RL simulation.  
The visualization will appear in **RViz**, directly accessible from the browser via the VNC interface.

---

## 🖥️ VNC Environment

The container uses the image `tiryoh/ros2-desktop-vnc:jazzy` to provide a browser-based development and visualization interface.  
This setup ensures **cross-platform reproducibility** and avoids dependency conflicts across operating systems.

If you prefer **not to use the VNC environment**, you can replace the base image in the Dockerfile with:
```
ros:jazzy
```
However, **this alone is not sufficient**. The `run.sh` script must also be modified accordingly to properly expose graphics devices and display parameters.  
These modifications depend on your system configuration (e.g., GPU type, graphics driver, operating system) and are therefore not detailed here.

---

## 🧠 Core Contribution

The original work developed in this repository lies entirely within the `ibvs_rl` package.  
It implements:
- The **IBVS controller** based on an approximated interaction matrix (ELM-PSO);
- An **adaptive gain tuning mechanism** using **Continuous Action Q-Learning (CAQL)**;
- Integration with ROS 2 for real-time simulation and control.

All other ROS packages serve as infrastructure to simulate the robot and sensors in Gazebo and MoveIt environments.

---

## 🎬 Video Tutorial

A step-by-step tutorial on how to build and run the simulation environment is available on YouTube:

👉 [Watch the setup video](https://youtu.be/LWE97ye5Ptg)

This video explains how to correctly build and launch the Docker-based ROS 2 environment and how to start the IBVS-RL simulation through the VNC interface.

---

## 📄 Thesis and Documentation

All theoretical background, design choices, and implementation details are thoroughly described in the Master’s Thesis document:

📘 **Full Thesis PDF (Google Drive):**  
[Online Reinforcement Learning for Adaptive Gain Tuning in Image-Based Visual Servoing](https://drive.google.com/file/d/139ezKAjrac02oMi4i4Ragl61CU84fFs2/view?usp=share_link)

The thesis explains the reasoning behind the architecture, control strategies, learning algorithms, and experimental setup implemented in this repository.

---

## 📚 Citation

If you use this work in academic or research projects, please cite:

> Nicola Saltarelli,  
> *Online Reinforcement Learning for Adaptive Gain Tuning in Image-Based Visual Servoing*,  
> Master’s Thesis, Politecnico di Bari, 2025.

---
