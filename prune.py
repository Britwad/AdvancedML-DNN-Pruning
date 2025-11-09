from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from Pruners.Pruners import SNIP, GraSP, SynFlow, Mag, Pruner

### other things to consider maybe 

def prune(model, device, prune_method, sample_dataloader, mask_type, sparsity  ):

    criterion = nn.CrossEntropyLoss()

    masked_parameters = []
    for m in model.modules():
        if hasattr(m, "weight_mask"):
            masked_parameters.append((m.weight_mask, m.weight))

    pruner = prune_method(masked_parameters)

    pruner.score(model, loss=criterion, dataloader=sample_dataloader, device=device)

    pruner.mask(sparsity, f'{mask_type}')

    pruner.apply_mask()

    print("# Trainable Parameters : " + str(pruner.stats()[0]))
    print("Total Amount of Parameters : " + str(pruner.stats()[1]))
    print("Sparsity % : " + str(pruner.stats()[0]/pruner.stats()[1]))
    return pruner.masked_parameters
