import torch



# mask loss
def mask_loss(target, reference, loss_type="l1"):
    if loss_type == "l1":
        return (target-reference).abs().mean()
    elif loss_type == "l2":
        return ((target-reference)**2).sum(-1).mean()
    else:
        raise NotImplementedError
    

# depth loss
def depth_loss(target, reference, mask, loss_type="l2"):

    # return (((((target - reference)* mask)**2).sum(-1)+1e-8)).sqrt().mean()
    if loss_type == "l1":
        return ((target-reference)*mask).abs().mean()
    elif loss_type == "l2":
        return (((target-reference)*mask)**2).sum(-1).mean()
    else:
        raise NotImplementedError
    

# normal loss
def normal_loss(target, reference, mask, loss_type="l2"):
    if loss_type == "l1":
        return ((target-reference)*mask).abs().mean()
    elif loss_type == "l2":
        return (((target-reference)*mask)**2).sum(-1).mean()
    else:
        raise NotImplementedError

# regularization loss
def regularization_loss(parameters, reg_type="l2"):
    if reg_type == "l2":
        return torch.mean(parameters**2)
    elif reg_type == "l1":
        return torch.mean(parameters.abs())
    else:
        raise NotImplementedError
    

# smooth loss
