import torch


def custom_collate_fn(batch):
    """Custom collate function for dataloader.
    Args:
        batch (tuple): list of batch_size tuples (X, Y), where:
            X (list): list of n_references + 1 tensors of shape (1, H, W)
            Y (list): list of n_references + 1 tensors of shape (3, H, W)
    Returns:
        batch_X_collated (torch.Tensor): tensor of shape (batch_size, 1, H, W)
        batch_Y_collated (torch.Tensor): tensor of shape (batch_size, 3, H, W)
    """
    X, Y = zip(*batch)
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)
    return X, Y