#配置代理
ssh -N -R 23234:127.0.0.1:23234 -p 55001 root@39.106.46.242

export http_proxy=http://127.0.0.1:23234
export https_proxy=http://127.0.0.1:23234

apt update

DEBIAN_FRONTEND=noninteractive apt install -y \
  git git-lfs build-essential cmake pkg-config \
  ffmpeg \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libswscale-dev libswresample-dev libavfilter-dev \
  software-properties-common

add-apt-repository -y ppa:deadsnakes/ppa
apt update

DEBIAN_FRONTEND=noninteractive apt install -y \
  python3.12 python3.12-venv python3.12-dev

python3.12 -m venv /opt/lerobot
source /opt/lerobot/bin/activate

python -m pip install -U pip setuptools wheel \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --no-cache-dir

  python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128

  cd /root/lerobot

pip install -e ".[pi]" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn

  pip install --force-reinstall "transformers==5.4.0"

  