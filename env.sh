# Source before any uv/vLLM command: keeps every cache, model and the venv under $OBS_ROOT
# (default ~/.observatorio; put large-disk overrides in the git-ignored env.local.sh).
[ -f "$(dirname "${BASH_SOURCE[0]}")/env.local.sh" ] && source "$(dirname "${BASH_SOURCE[0]}")/env.local.sh"
export OBS_ROOT=${OBS_ROOT:-$HOME/.observatorio}
export HF_HOME=$OBS_ROOT/hf
export UV_CACHE_DIR=$OBS_ROOT/uv-cache
export UV_PYTHON_INSTALL_DIR=$OBS_ROOT/python
export UV_PROJECT_ENVIRONMENT=$OBS_ROOT/venv
export OBS_DATA=$OBS_ROOT/data
export VLLM_CACHE_ROOT=$OBS_ROOT/vllm-cache
export TRITON_CACHE_DIR=$OBS_ROOT/triton-cache
