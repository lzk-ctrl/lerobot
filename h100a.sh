
# Run the ROS2 bridge to receive cmd_vel messages and send them to the robot
conda deactivate
source /opt/ros/humble/setup.bash
source /home/lzk/eai_ws/install/setup.bash
python3 /root/lerobot/examples/tutorial/pi05/ros2_cmd_vel_bridge.py --cmd-vel-topic /cmd_vel --udp-port 5555 --verbose

#pi05 inference server
python examples/tutorial/pi05/pi05_inference_server.py \
    --model-id /root/models/pi05_base   \
    --host 127.0.0.1 \
    --port 6001 \
    --chunk-size 20

#pi05 client to send tasks and receive cmd_vel commands
python examples/tutorial/pi05/call_pi05_server.py     --server-host 127.0.0.1     --server-port 6001     --snapshot-server-url http://127.0.0.1:7001/latest     --continuous     --send-mode first     --actions-per-chunk 1     --execution-mode task_execution     --execution-horizon-sec 0.1     --execution-settle-sec 0.05     --task "walk forward while avoiding all obstacles"     --cmd-vel-output print+udp     --udp-host 127.0.0.1     --udp-port 5555

#get observation snapshots for visualization
bash examples/tutorial/pi05/run_observation_snapshot_server.sh --host 127.0.0.1 --port 7001

python examples/tutorial/pi05/pi05_inference_server.py --model-id /root/models/pi05_base   --host 127.0.0.1 --port 6002 --device cuda:1 --num-inference-steps 10 

python examples/tutorial/pi05/call_pi05_server.py \
    --server-host 127.0.0.1 \
    --server-port 6001 \
    --observation-json /root/lerobot/observation_with_image.json \
    --send-mode first \
    --actions-per-chunk 1 \
    --execution-mode continuous_control \
    --cmd-vel-output print 
    --save-final-noise-path /root/lerobot/tmp/pi05_noise_state.pt

python examples/tutorial/pi05/call_pi05_server.py \
    --server-host 127.0.0.1 \
    --server-port 6002 \
    --observation-json /root/lerobot/observation_with_image.json \
    --send-mode first \
    --actions-per-chunk 1 \
    --execution-mode continuous_control \
    --cmd-vel-output print \
    --initial-noise-path /root/lerobot/tmp/pi05_noise_state.pt

python examples/tutorial/pi05/call_pi05_server.py \
    --server-host 127.0.0.1 \
    --server-port 6002 \
    --observation-json /root/lerobot/observation_with_image_variant_large_shift.json \
    --send-mode first \
    --actions-per-chunk 1 \
    --execution-mode continuous_control \
    --cmd-vel-output print \
    --initial-noise-path /root/lerobot/tmp/pi05_noise_state.pt

 python examples/tutorial/pi05/compare_pi05_action_variance.py \
    --server-host 127.0.0.1 \
    --server-port 6002 \
    --observation-json /root/lerobot/observation_with_image_variant_large_shift.json \
    --actions-per-chunk 10 --runs 20
    
sudo -E env      PATH=/home/lzk/anaconda3/envs/lerobot/bin:$PATH      OUTPUT_BASE=/root/lerobot/tmp/pi05_nsys_n4      WARMUP_RUNS=3      PROFILE_RUNS=1      STEPS=10      NUM_ACTION_SAMPLES=4      DEVICE=cuda:1      OBSERVATION_JSON=/root/lerobot/observation_with_image.json      ./examples/tutorial/pi05/profile_pi05_nsys.sh  #nsys

  python examples/tutorial/pi05/call_pi05_server_fixed_rate.py \
    --rate-hz 3 \
    --server-host 127.0.0.1 \
    --server-port 6001 \
    --observation-json /root/lerobot/observation_with_image.json \
    --send-mode first \
    --actions-per-chunk 1 \
    --execution-mode continuous_control \
    --cmd-vel-output print

source /opt/lerobot/bin/activate