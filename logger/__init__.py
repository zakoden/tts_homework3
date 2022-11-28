from ..configs import train_config
from .wandb_writer import WanDBWriter

logger = WanDBWriter(train_config)
