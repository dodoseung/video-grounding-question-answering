import pickle
import time

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    """Whether torch.distributed is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """Number of processes in the current process group."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Rank of the current process."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """True if current process is rank 0."""
    return get_rank() == 0


def synchronize() -> None:
    """Synchronize all processes."""
    if not is_dist_avail_and_initialized():
        return
    if get_world_size() == 1:
        return
    dist.barrier()


def all_gather(data):
    """Gather arbitrary picklable data from all ranks.

    Returns a list of data gathered from each rank, with the local data in the
    position corresponding to the local rank.
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # serialize to tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device)

    # obtain sizes of each rank
    local_size = torch.tensor([tensor.numel()], dtype=torch.long, device=device)
    size_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(sz.item()) for sz in size_list]
    max_size = max(size_list)

    # receive tensors from all ranks
    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=device) for _ in size_list]
    if local_size.item() != max_size:
        padding = torch.empty(size=(max_size - local_size.item(),), dtype=torch.uint8, device=device)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for sz, t in zip(size_list, tensor_list):
        buf = t.cpu().numpy().tobytes()[:sz]
        data_list.append(pickle.loads(buf))
    return data_list


def reduce_dict(input_dict, average: bool = True):
    """Reduce dictionary values across all processes with optional averaging."""
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if get_rank() == 0 and average:
            values /= world_size
        return {k: v for k, v in zip(names, values)}


def reduce_loss_dict(loss_dict):
    """Reduce loss dictionary from all processes and average on rank 0."""
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if get_rank() == 0:
            all_losses /= world_size
        return {k: v for k, v in zip(loss_names, all_losses)}


