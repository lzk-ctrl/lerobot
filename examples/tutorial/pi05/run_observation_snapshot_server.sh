#!/usr/bin/env bash

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

set +u
source /opt/ros/humble/setup.bash
source "${HOME}/eai_ws/install/setup.bash"
set -u

export ROS_HOME="${REPO_ROOT}/.ros_home"
export ROS_LOG_DIR="${REPO_ROOT}/.ros_logs"
mkdir -p "${ROS_HOME}" "${ROS_LOG_DIR}"

exec /usr/bin/python3 "${REPO_ROOT}/examples/tutorial/pi05/observation_snapshot_server.py" "$@"
