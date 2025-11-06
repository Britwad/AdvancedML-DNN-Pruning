from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from Pruners.Pruners import SNIP, GraSP, SynFlow, Mag, Pruner

### other things to consider maybe 

def prune(net, device, prune_method, sample_dataloader, mask_type, sparsity  ):

    model = net().to(device)
    criterion = nn.CrossEntropyLoss()

    masked_parameters = []
    for m in model.modules():
        if hasattr(m, "weight_mask"):
            masked_parameters.append((m.weight_mask, m.weight))

    pruner = prune_method(masked_parameters)

    pruner.score(model, loss=criterion, dataloader=sample_dataloader, device=device)

    pruner.mask(sparsity, f'{mask_type}')

    pruner.apply_mask()

    return pruner.masked_parameters
