"""Observation packing for replay storage and the reservoir buffer used by Deep CFR."""
import numpy as np

from bridge.env.bridge_env import OBS_SIZE, TRICKS

_TRICK_IDX = slice(TRICKS, TRICKS + 2)
_PACKED_BIN_LEN = (OBS_SIZE + 7) // 8


def pack_obs(obs):
    obs = np.asarray(obs, dtype=np.float32)
    bits = np.packbits((obs > 0.5).astype(np.uint8))
    tricks = np.rint(obs[_TRICK_IDX] * 13.0).astype(np.uint8)
    return np.concatenate([bits, tricks])


def unpack_obs_batch(packed_list):
    arr = np.asarray(packed_list, dtype=np.uint8)
    if arr.ndim == 1:
        arr = arr[None, :]
    out = np.unpackbits(arr[:, :_PACKED_BIN_LEN], axis=1)[:, :OBS_SIZE].astype(np.float32)
    out[:, _TRICK_IDX] = arr[:, _PACKED_BIN_LEN:].astype(np.float32) / 13.0
    return out


class ReservoirBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.num_seen = 0

    def add(self, item):
        self.num_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            j = np.random.randint(0, self.num_seen)
            if j < self.capacity:
                self.buffer[j] = item

    def sample(self, batch_size):
        n = len(self.buffer)
        if n == 0:
            return []
        idx = np.random.randint(0, n, size=min(batch_size, n))
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)
