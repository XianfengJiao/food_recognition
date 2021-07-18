import torch
import numpy as np
import torch.nn as nn

from args import get_parser

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# # =============================================================================

class Triplet(nn.Module):

    def __init__(self):
        super(Triplet, self).__init__()
        self.margin = opts.margin
        self.id_background = 0

        self.sem_weight = opts.sem_weight
        self.cos_weight = opts.cos_weight

        self.nb_samples = 9999

    def calculate_cost(self, cost):
        ans, _ = torch.sort(cost, dim=1, descending=True)          
        return ans[:,:self.nb_samples]

    def add_cost(self, name, cost):
        invalid_pairs = (cost == 0).float().sum()

        valid_pairs = cost.numel() - invalid_pairs
        if name in ['IRR', 'RII']:
            weight = self.cos_weight
        elif name in ['SIRR', 'SRII']:
            weight = self.sem_weight

        return cost.sum() * weight / valid_pairs

    def semantic_multimodal(self, distances, class1, class2, erase_diagonal=True):
        class1_matrix = class1.repeat(class1.size(0), 1)
        class2_matrix = class2.repeat(class2.size(0), 1).t()

        matrix_mask = ((class1_matrix != 0) + (class2_matrix != 0)) == 2

        same_class = torch.eq(class1_matrix, class2_matrix)
        anti_class = same_class.clone()
        anti_class = anti_class == 0 # get the dissimilar classes

        if erase_diagonal:
            same_class[range(same_class.size(0)),range(same_class.size(1))] = 0 # erase instance-instance pairs
        new_dimension = matrix_mask.int().sum(1).max().item()
        same_class = torch.masked_select(same_class, matrix_mask).view(new_dimension, new_dimension)
        anti_class = torch.masked_select(anti_class, matrix_mask).view(new_dimension, new_dimension)
        mdistances = torch.masked_select(distances, matrix_mask).view(new_dimension, new_dimension)

        same_class[same_class.cumsum(dim=1) > 1] = 0 # erasing extra positives
        pos_samples = torch.masked_select(mdistances, same_class) # only the first one
        min_neg_samples = anti_class.int().sum(1).min().item() # selecting max negatives possible
        anti_class[anti_class.cumsum(dim=1) > min_neg_samples] = 0 # erasing extra negatives
        neg_samples = torch.masked_select(mdistances, anti_class).view(new_dimension, min_neg_samples)

        cost = pos_samples.unsqueeze(1) - neg_samples + self.margin
        cost[cost < 0] = 0 # hinge
        return cost

    def __call__(self, input1, input2, class1, class2):

        loss = input1.data.new([0])

        # Detect and treat unbalanced batch
        size1 = class1.size(0)
        size2 = class2.size(0)
        assert size1 == size2
        
        # Instance-based triplets
        distances = self.dist(input1, input2)
        #'IRR'
        cost = distances.diag().unsqueeze(1) - distances + self.margin # all triplets
        cost[cost < 0] = 0 # hinge
        cost[range(cost.size(0)),range(cost.size(1))] = 0 # erase pos-pos pairs
        loss += self.add_cost('IRR', self.calculate_cost(cost))
        #'RII'
        cost = distances.diag().unsqueeze(0) - distances + self.margin # all triplets
        cost[cost < 0] = 0 # hinge
        cost[range(cost.size(0)),range(cost.size(1))] = 0 # erase pos-pos pairs
        loss += self.add_cost('RII', self.calculate_cost(cost.t()))


        # Prepare semantic samples (class != 0)
        valid_input1 = class1 != 0
        valid_input2 = class2 != 0
        semantic_input1 = input1[valid_input1].view(valid_input1.sum().int().item(), input1.size(1))
        semantic_class1 = class1[valid_input1]
        semantic_input2 = input2[valid_input2].view(valid_input2.sum().int().item(), input2.size(1))
        semantic_class2 = class2[valid_input2]    
        # Semantic-based triplets
        distances = self.dist(semantic_input1, semantic_input2)
        # 'SIRR'
        cost = self.semantic_multimodal(distances, semantic_class1, semantic_class2)
        loss += self.add_cost('SIRR', self.calculate_cost(cost))
        #'SRII'
        cost = self.semantic_multimodal(distances.t(), semantic_class2, semantic_class1)
        loss += self.add_cost('SRII', self.calculate_cost(cost))

        return loss

    def dist(self, input_1, input_2):
        #assert both input1 and input2 have been normalized
        return 1 - torch.mm(input_1, input_2.t())
