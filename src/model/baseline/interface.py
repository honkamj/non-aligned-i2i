"""Interface to baseline methods"""

from typing import Optional
from torch import device, load, save
from torch.nn import Module
from torch.optim import Optimizer
from util.training import get_model_save_path

def save_reggan_model(
    target_dir: str,
    epoch: int,
    prefix: str,
    generator_model: Module,
    discriminator_model: Module,
    reg_model: Module,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    reg_optimizer: Optimizer,
) -> None:
    """Save Reg GAN model"""
    save_dict = {
        "epoch": epoch,
        "generator_model_state_dict": generator_model.state_dict(),
        "discriminator_model_state_dict": discriminator_model.state_dict(),
        "reg_model_state_dict": reg_model.state_dict(),
        "generator_optimizer_state_dict": generator_optimizer.state_dict(),
        "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(),
        "reg_optimizer_state_dict": reg_optimizer.state_dict(),
    }
    save(save_dict, get_model_save_path(target_dir, epoch, prefix))


def load_reggan_model(
    target_dir: str,
    epoch: int,
    prefix: str,
    torch_device: device,
    generator_model: Module,
    discriminator_model: Optional[Module] = None,
    reg_model: Optional[Module] = None,
    generator_optimizer: Optional[Optimizer] = None,
    discriminator_optimizer: Optional[Optimizer] = None,
    reg_optimizer: Optional[Optimizer] = None,
) -> int:
    """Load Reg GAN model"""
    checkpoint = load(get_model_save_path(target_dir, epoch, prefix), map_location=torch_device)
    generator_model.load_state_dict(checkpoint["generator_model_state_dict"])
    if discriminator_model is not None:
        discriminator_model.load_state_dict(checkpoint["discriminator_model_state_dict"])
    if reg_model is not None:
        reg_model.load_state_dict(checkpoint["reg_model_state_dict"])
    if generator_optimizer is not None:
        generator_optimizer.load_state_dict(checkpoint["generator_optimizer_state_dict"])
    if discriminator_optimizer is not None:
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])
    if reg_optimizer is not None:
        reg_optimizer.load_state_dict(checkpoint["reg_optimizer_state_dict"])
    return checkpoint["epoch"]
