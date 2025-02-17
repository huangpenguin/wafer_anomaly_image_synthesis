from anomalib.models import Draem
from omegaconf import OmegaConf

from .patchcore import PatchcoreDetector
from .reverse_distillation import ReverseDistillationDetector
from .timm import TimmDetector
from .vae import VAEDetector

__all__ = [
    "PatchcoreDetector",
    "ReverseDistillationDetector",
    "VAEDetector",
    "TimmDetector"
]

def get_model(model_name: str, **kwargs):
    if "model_params" in kwargs:
        kwargs = kwargs | OmegaConf.to_container(kwargs["model_params"])
        
    if model_name == "patchcore":
        Model = PatchcoreDetector
    elif model_name == "draem":
        Model = Draem
    elif model_name == "rd4ad":
        Model = ReverseDistillationDetector
    elif model_name == "vae":
        Model = VAEDetector
    elif model_name == "resnet50":
        Model = TimmDetector
    else:
        raise RuntimeError("Specified model does not exists in the model list.")
    kwargs = kwargs | {"model_name": model_name}
    
    if "checkpoint_path" in kwargs:
        return Model.load_from_checkpoint(**kwargs)
    else:
        return Model(**kwargs)