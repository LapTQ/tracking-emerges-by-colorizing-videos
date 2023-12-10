import importlib


def backbone_factory(
        **kwargs
):
    # parse kwargs
    module_name = kwargs['module_name']
    
    module = importlib.import_module(
        name='{}.{}'.format(__name__, module_name)
    )
    cls_ = getattr(module, 'CustomBackbone')

    return cls_