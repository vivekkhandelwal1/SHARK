TD="$(cd $(dirname $0) && pwd)"
if [ -z "$PYTHON" ]; then
  PYTHON="$(which python3)"
fi

function die() {
  echo "Error executing command: $*"
  exit 1
}

PYTHON_VERSION_X_Y=`${PYTHON} -c 'import sys; version=sys.version_info[:2]; print("{0}.{1}".format(*version))'`

echo "Python: $PYTHON"
echo "Python version: $PYTHON_VERSION_X_Y"

pip install ftfy tqdm regex
pip install git+https://github.com/openai/CLIP.git
pip uninstall -y torch torchvision
pip install -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html --pre torch torchvision

mkdir v-diffusion-pytorch/checkpoints
# wget https://the-eye.eu/public/AI/models/v-diffusion/cc12m_1_cfg.pth -P v-diffusion-pytorch/checkpoints/
