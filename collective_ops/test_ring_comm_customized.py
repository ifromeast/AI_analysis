from typing import Optional
import torch
import torch.distributed as dist


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(
            dist.isend, to_send, self.send_rank, group=self._process_group
        )
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []


# 初始化进程组
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device("cuda:{}".format(rank))

ring_comm = RingComm(process_group=None)

tensor = torch.tensor([rank, rank+1], dtype=torch.float32, device=device)
print("rank {} tensor {}".format(rank, tensor))

recv_tensor = ring_comm.send_recv(tensor)
# Commit the operations
ring_comm.commit()
# Wait for the operations to complete
ring_comm.wait()

print(f"Rank {rank} received: {recv_tensor}")


# 销毁进程组
dist.destroy_process_group()


    
  

