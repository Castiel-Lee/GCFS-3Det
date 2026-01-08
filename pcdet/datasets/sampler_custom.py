import torch
import numpy as np
from torch.utils.data.sampler import Sampler
import random

class Sampler_CL(Sampler):
    def __init__(self, dataset, basic_size):
        self.dataset = dataset
        self.basic_size = dataset.__len__()

    def __iter__(self):
        indices = list(range(self.basic_size))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.basic_size

class BatchSampler_CL:
    def __init__(self, sampler, batch_size, basic_size, extra_batch_size, drop_last=True):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.basic_size = basic_size
        self.extra_len = extra_batch_size

    def __iter__(self):
        batch = []
        i = 0
        sampler_list = list(self.sampler)
        for idx in sampler_list:
            batch.append(idx)
            if len(batch) >= self.batch_size:  
                batch += [random.randint(1, self.basic_size-1) for _ in range(self.extra_len)]  
                yield batch
                batch = []
            i += 1
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size