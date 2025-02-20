import logging
from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def compute_num_params(model):
    """Get num params."""
    num_params = 0
    seen = set()
    for p in model.parameters():  # pylint: disable=invalid-name
        if p not in seen:
            seen.add(p)
            if hasattr(p, "ds_shape"):
                num_params += np.prod(p.ds_shape)
            else:
                num_params += np.prod(p.size())

    return num_params

_logger = None
def get_logger():
    global _logger
    if _logger is None:
        logging.getLogger("torch.distributed.checkpoint._dedup_tensors").setLevel(logging.ERROR)
        logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.ERROR)
        _logger = logging.getLogger(__name__)
        _logger.setLevel(logging.INFO)
        _logger.handlers = []
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname).1s " "[%(filename)s:%(lineno)d] %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        _logger.addHandler(ch)
        _logger.propagate = False
    return _logger

