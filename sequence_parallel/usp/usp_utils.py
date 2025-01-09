import torch
import torch.distributed as dist

class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance


class ProcessGroupSingleton(Singleton):
    def __init__(self):
        self.ULYSSES_PG = None
        self.RING_PG = None


PROCESS_GROUP = ProcessGroupSingleton()

def set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, world_size, use_ulysses_low=True):
    """
    sp_ulysses_degree x sp_ring_degree = seq_parallel_degree
    (ulysses_degree, dp_degree)
    """
    sp_degree = sp_ring_degree * sp_ulysses_degree
    dp_degree = world_size // sp_degree

    assert (world_size % sp_degree == 0), f"world_size {world_size} % sp_degree {sp_ulysses_degree} == 0"

    num_ulysses_pgs = sp_ring_degree  # world_size // sp_ulysses_degree
    num_ring_pgs = sp_ulysses_degree  # world_size // sp_ring_degree

    if use_ulysses_low:
        for dp_rank in range(dp_degree):
            offset = dp_rank * sp_degree
            for i in range(num_ulysses_pgs):
                ulysses_ranks = list(range(i * sp_ulysses_degree + offset, (i + 1) * sp_ulysses_degree + offset,))
                group = torch.distributed.new_group(ulysses_ranks)
                if rank in ulysses_ranks:
                    ulyssess_pg = group

            for i in range(num_ring_pgs):
                ring_ranks = list(range(i + offset, sp_degree + offset, num_ring_pgs))
                group = torch.distributed.new_group(ring_ranks)
                if rank in ring_ranks:
                    ring_pg = group

    else:
        for dp_rank in range(dp_degree):
            offset = dp_rank * sp_degree
            for i in range(num_ring_pgs):
                ring_ranks = list(
                    range(
                        i * sp_ring_degree + offset, (i + 1) * sp_ring_degree + offset
                    )
                )
                group = torch.distributed.new_group(ring_ranks)
                if rank in ring_ranks:
                    ring_pg = group

            for i in range(num_ulysses_pgs):
                ulysses_ranks = list(
                    range(i + offset, sp_degree + offset, num_ulysses_pgs)
                )
                group = torch.distributed.new_group(ulysses_ranks)
                if rank in ulysses_ranks:
                    ulyssess_pg = group

    PROCESS_GROUP.ULYSSES_PG = ulyssess_pg
    PROCESS_GROUP.RING_PG = ring_pg


def stripe_extract_local(value, rank, world_size, rd, ud, *args, **kwargs):
    # ud at the highest dim
    input_dim = value.dim()
    assert input_dim >= 2

    batch_size, seqlen, *rest = value.shape

    assert dist.get_world_size(group=PROCESS_GROUP.RING_PG) == rd
    assert dist.get_world_size(group=PROCESS_GROUP.ULYSSES_PG) == ud
    
    value = value.reshape(batch_size, seqlen // rd, rd, -1).contiguous()
    value = value.transpose(1, 2).reshape(batch_size, seqlen, -1).contiguous()
    value = value.chunk(world_size, dim=1)[rank]

    new_shape = [batch_size, seqlen // world_size] + rest
    return value.reshape(new_shape)


def basic_extract_local(value, rank, world_size, *args, **kwargs):
    return value.chunk(world_size, dim=1)[rank].detach().clone()


def zigzag_extract_local(value, rank, world_size, rd, ud, dim=1, *args, **kwargs):
    """
    value is a tensor of shape (bs, seqlen, ...)
    """
    input_dim = value.dim()
    assert input_dim >= 2
    batch_size, seqlen, *rest = value.shape

    value_chunks = value.chunk(2 * rd, dim=dim)
    r_rank = dist.get_rank(group=PROCESS_GROUP.RING_PG)
    u_rank = dist.get_rank(group=PROCESS_GROUP.ULYSSES_PG)

    assert dist.get_world_size(group=PROCESS_GROUP.RING_PG) == rd
    assert dist.get_world_size(group=PROCESS_GROUP.ULYSSES_PG) == ud

    local_value = torch.cat([value_chunks[r_rank], value_chunks[2 * rd - r_rank - 1]], dim=dim).chunk(ud, dim=dim)[u_rank]

    new_shape = [batch_size, seqlen // world_size] + rest
    return local_value.reshape(new_shape).contiguous()


EXTRACT_FUNC_DICT = {
    "basic": basic_extract_local,
    "stripe": stripe_extract_local,
    "zigzag": zigzag_extract_local
}


# test if flash_attn is available
try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    from flash_attn_interface import _flash_attn_forward as flash_attn_forward_hopper
    from flash_attn_interface import _flash_attn_backward as flash_attn_func_hopper_backward
    from flash_attn_interface import flash_attn_func as flash3_attn_func
    HAS_FLASH_ATTN_HOPPER = True
except ImportError:
    HAS_FLASH_ATTN_HOPPER = False