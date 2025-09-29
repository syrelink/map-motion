import torch
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch was built with CUDA version: {torch.version.cuda}")
print(f"Is CUDA available for PyTorch? {torch.cuda.is_available()}")