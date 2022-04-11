from typing import List


class DecayStrategy:
    def __init__(self, total_epoch: int, decay_items: List[str] = list(), enable: bool = True) -> None:
        self.total_epoch = total_epoch
        self.enable = enable
        self.decay_items = decay_items

    def __call__(self, epoch: int, prefix: str = None):
        decay = 1.0
        flag = self.enable
        flag = flag and (prefix is None or prefix in self.decay_items)
        if flag:
            decay = 1.0 - epoch / self.total_epoch
            if decay < 0:
                decay = 0
        return decay

