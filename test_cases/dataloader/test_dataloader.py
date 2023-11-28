from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent

sys.path.append(str(ROOT_DIR))

# ==================================================================================================

import src as GLOBAL
from src.dataloader import load_module

DATALOADER_CONFIG = GLOBAL.CONFIG['dataloader']

dataloader = load_module(
    module_name=DATALOADER_CONFIG['module_name']
)(
    **DATALOADER_CONFIG['kwargs']
)

input_, label = next(iter(dataloader))
print(input_)



