import torch


def gpu_info():
    print(f"Is available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device name: {torch.cuda.get_device_name(0)}")
    print(f"Current device properties: {torch.cuda.get_device_properties(0)}")
    print(f"Current device capabilities: {torch.cuda.get_device_capability(0)}")
