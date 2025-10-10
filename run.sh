docker run -it --rm --ipc host \
    -p 6080:80 \
    -e DISPLAY=:1 \
    -v ./ros_ws/:/root/ros_workspace \
    --name ibvs_rl \
    ros_jazzy:ibvs_rl bash
    