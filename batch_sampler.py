import copy
import torch
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.sampler import BatchSampler


class RandomSamplerValues(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        generator = iter(torch.randperm(len(self.data_source)).long())
        for value in generator:
            yield self.data_source[value]

    def __len__(self):
        return len(self.data_source)


class SemanticBatchSampler(object):
    def __init__(self, indices_by_class, batch_size, same_class_num):

        assert batch_size % same_class_num == 0

        self.indices_by_class = indices_by_class
        self.batch_size = batch_size
        self.same_class_num = same_class_num

        self.batch_sampler_by_class = []
        for indices in indices_by_class:
            self.batch_sampler_by_class.append(
                BatchSampler(RandomSamplerValues(indices),
                             self.same_class_num,
                             True))

    def sampler_lengths(self):
        return [len(sampler) for sampler in self.batch_sampler_by_class]
        
    def __iter__(self):
        
        sampler_lens = torch.Tensor(self.sampler_lengths())
        gen_by_class = [sampler.__iter__() for sampler in self.batch_sampler_by_class]

        for i in range(len(self)):
            batch = []
            nb_samples = self.batch_size // self.same_class_num
            for j in range(nb_samples):
                # Class sampling

                idx = torch.multinomial(sampler_lens,
                        1, # num_samples
                        False)[0] #replacement

                sampler_lens[idx] -= 1
                batch += gen_by_class[idx].__next__()
            yield batch

    def __len__(self):
        return sum(self.sampler_lengths()) // (self.batch_size//self.same_class_num)


class BatchSamplerTriplet(object):

    def __init__(self, indices_by_class, batch_size, semantic_pc=0.5, same_class_num=2):
        self.semantic_indices = copy.deepcopy(indices_by_class)
        self.background_indices = self.semantic_indices.pop(0)
        self.batch_size = batch_size
        self.semantic_pc = semantic_pc
        self.same_class_num = same_class_num

        self.background_size = round((1 - self.semantic_pc) * self.batch_size)
        self.semantic_size = self.batch_size - self.background_size

        # Batch Sampler NoClassif
        self.background_sampler = BatchSampler(
            RandomSamplerValues(self.background_indices),
            self.background_size,
            True)

        # Batch Sampler Classif
        self.semantic_sampler = SemanticBatchSampler(
            self.semantic_indices,
            self.semantic_size,
            self.same_class_num)

    def __iter__(self):
        gen_semantic = self.semantic_sampler.__iter__()
        gen_background = self.background_sampler.__iter__()
        for i in range(len(self)):
            batch = []
            batch += gen_semantic.__next__()
            batch += gen_background.__next__()
            yield batch

    def __len__(self):
        return min([len(self.semantic_sampler),
                    len(self.background_sampler)])
