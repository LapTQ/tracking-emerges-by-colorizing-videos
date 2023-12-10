from torch import nn


class CustomHead(nn.Module):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__()

        # parse kwargs
        self.n_references = kwargs['n_references']

        return

    def forward(
            self,
            x
    ):
        return x