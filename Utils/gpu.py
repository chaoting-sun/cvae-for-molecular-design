import torch

def allocate_gpu(id=None, n_total_gpus=4):
    v = torch.empty(1)
    if id is not None:
        return torch.device("cuda:{}".format(str(id)))
    else:
        for i in range(n_total_gpus):
            try:
                dev_name = "cuda:{}".format(str(i))
                v = v.to(dev_name)
                print("Allocating cuda:{}.".format(i))
                return v.device
            except Exception as e:
                pass
        exit("CUDA error: all CUDA-capable devices are busy or unavailable")
        return v.device