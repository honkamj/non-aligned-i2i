"""Training utilities"""

from argparse import ArgumentParser
from json import dump as json_dump
from json import load as json_load
from os import listdir, makedirs
from os.path import basename, isfile, join
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from git import Repo  # type: ignore
from torch import device, load, no_grad, save
from torch.nn import Module
from torch.optim import Optimizer


class LossAverager:
    """Utility class for averaging over individual loss outputs"""

    def __init__(self) -> None:
        self._masses: Dict[str, float] = {}
        self._loss_sums: Dict[str, float] = {}
        self._mass_functions: Dict[str, Callable[[float, float], float]] = {}

    def count(self, losses: Mapping[str, float], mass: float = 1.0) -> None:
        """Add loss value to average"""
        for loss_name, loss_value in losses.items():
            self._masses[loss_name] = self._masses.get(loss_name, 0) + mass
            self._loss_sums[loss_name] = self._loss_sums.get(loss_name, 0.0) + loss_value

    def set_custom_mass_func(self, loss: str, mass_func: Callable[[float, float], float]) -> None:
        """Define custom mass function (instead of average)"""
        self._mass_functions[loss] = mass_func

    def save_to_json(self, epoch: int, filename: str, postfix: str = "") -> None:
        """Save agregated losses to json-file"""
        if isfile(filename):
            with open(filename, mode="r", encoding="UTF-8") as loss_file_read:
                loss_dict = json_load(loss_file_read)
        else:
            loss_dict = {}
        losses = self._get_losses()
        losses = {f"{loss_name}{postfix}": loss_value for loss_name, loss_value in losses.items()}
        loss_dict[str(epoch)] = losses
        with open(filename, mode="w", encoding="UTF-8") as loss_file_write:
            json_dump(loss_dict, loss_file_write, indent=4)

    def save_state(self, filename: str) -> None:
        """Save state to json-file"""
        state = {}
        state["masses"] = self._masses
        state["loss_sums"] = self._loss_sums
        with open(filename, mode="w", encoding="UTF-8") as state_file:
            json_dump(state, state_file, indent=4)

    def load_state(self, filename: str) -> None:
        """Load state from file"""
        with open(filename, mode="r", encoding="UTF-8") as state_file:
            state = json_load(state_file)
        for loss_name, loss_mass in state["masses"].items():
            if loss_name not in self._masses:
                self._masses[loss_name] = 0.0
            self._masses[loss_name] += loss_mass
        for loss_name, loss_sum in state["loss_sums"].items():
            if loss_name not in self._loss_sums:
                self._loss_sums[loss_name] = 0.0
            self._loss_sums[loss_name] += loss_sum

    def _get_losses(self) -> Mapping[str, float]:
        return {
            loss_name: self._mass_functions.get(loss_name, self._average)(
                loss_sum, self._masses[loss_name]
            )
            for loss_name, loss_sum in self._loss_sums.items()
        }

    def __repr__(self) -> str:
        return ", ".join(
            f"{loss_name}={loss_value:.4}" for loss_name, loss_value in self._get_losses().items()
        )

    @staticmethod
    def _average(loss_sum: float, mass: float) -> float:
        if mass == 0:
            return float("nan")
        return loss_sum / mass


def get_model_save_path(target_dir: str, epoch: int, prefix: str) -> str:
    """Get target path for saving a model"""
    return join(target_dir, f"{prefix}_weights_epoch_{int(epoch):03d}.pt")


def save_model(
    target_dir: str,
    epoch: int,
    prefix: str,
    model: Module,
    discriminator_model: Optional[Module] = None,
    optimizer: Optional[Optimizer] = None,
    discriminator_optimizer: Optional[Optimizer] = None,
) -> None:
    """Save model for continuing training later"""
    save_dict = {"epoch": epoch, "model_state_dict": model.state_dict()}
    if discriminator_model is not None:
        save_dict.update({"discriminator_model_state_dict": discriminator_model.state_dict()})
    if optimizer is not None:
        save_dict.update({"optimizer_state_dict": optimizer.state_dict()})
    if discriminator_optimizer is not None:
        save_dict.update(
            {"discriminator_optimizer_state_dict": discriminator_optimizer.state_dict()}
        )
    save(save_dict, get_model_save_path(target_dir, epoch, prefix))


def get_commit_hash() -> str:
    """Return current commit hash of the git HEAD"""
    return Repo(search_parent_directories=True).head.object.hexsha


def load_model(
    target_dir: str,
    epoch: int,
    prefix: str,
    model: Module,
    torch_device: device,
    discriminator_model: Optional[Module] = None,
    optimizer: Optional[Optimizer] = None,
    discriminator_optimizer: Optional[Optimizer] = None,
) -> int:
    """Load model for continuing training"""
    checkpoint = load(get_model_save_path(target_dir, epoch, prefix), map_location=torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if discriminator_model is not None:
        discriminator_model.load_state_dict(checkpoint["discriminator_model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if discriminator_optimizer is not None:
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])
    return checkpoint["epoch"]


def _get_filename(sample: str, postfix: str) -> str:
    return f"{sample}_{postfix}"


def get_path(data_root: str, postfix: str, sample: str) -> str:
    """Obtain path to data file in input data structure"""
    return join(data_root, sample, _get_filename(basename(sample), postfix))


def find_largest_epoch(model_root_dir: str, prefix: str) -> Optional[int]:
    """Obtain paths to all samples of each file type"""
    largest_epoch: Optional[int] = None
    total_prefix = f"{prefix}_weights_epoch_"
    prefix_length = len(total_prefix)
    for file in listdir(model_root_dir):
        if file.endswith(".pt") and file.startswith(total_prefix):
            epoch = int(file[prefix_length:-3])
            if largest_epoch is None or epoch > largest_epoch:
                largest_epoch = epoch
    return largest_epoch


TrainingFunction = Callable[
    [
        Mapping[str, Any],  # config
        str,  # target directory
        int,  # seed
        Optional[int],  # continue from epoch
        Sequence[device],  # devices to use
    ],
    None,
]


def obtain_arguments_and_train(training_function: TrainingFunction) -> None:
    """Parse arguments for training and train the model"""
    parser = ArgumentParser()
    parser.add_argument("--config", help="Path to config file", type=str, required=False)
    parser.add_argument("--target-root", help="Path to output root", type=str, required=True)
    parser.add_argument(
        "--model-name", help="Model name used in saving the model", type=str, required=True
    )
    parser.add_argument(
        "--continue-from-epoch",
        help="Training is continued from epoch after this",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--devices", help="Names of the devices to use", type=str, nargs="+", required=True
    )
    parser.add_argument(
        "--seed", help="Seed to use for generating the dataset", type=int, required=True
    )
    args = parser.parse_args()
    target_dir = join(args.target_root, args.model_name)
    if args.config is None:
        config_path = join(target_dir, "training_config.json")
    else:
        config_path = args.config
    with open(config_path, mode="r", encoding="UTF-8") as config_file:
        config = json_load(config_file)
    makedirs(target_dir, exist_ok=True)
    if args.continue_from_epoch is None:
        continue_from_epoch = find_largest_epoch(target_dir, "training")
    else:
        continue_from_epoch = args.continue_from_epoch
    with open(
        join(target_dir, "training_config.json"), mode="w", encoding="UTF-8"
    ) as config_copy_file:
        config["commit"] = get_commit_hash()
        config["seed"] = args.seed
        json_dump(config, config_copy_file, indent=4)
    devices = [device(device_name) for device_name in args.devices]
    training_function(config, target_dir, args.seed, continue_from_epoch, devices)


EvaluatingFunction = Callable[
    [
        Mapping[str, Any],  # config
        str,  # target directory
        int,  # seed
        Iterable[int],  # epochs to evaluate
        bool,  # wheter to plot outputs
        Optional[int],  # sequence item
        bool,  # shuffle dataset
        str,  # dataset to use (train, test, validate)
        bool,  # Whether to use only patches with full mask
        bool,  # Whether to skip evaluated
        Sequence[device],  # devices to use
    ],
    None,
]


def obtain_arguments_and_evaluate(evaluating_function: EvaluatingFunction) -> None:
    """Parse arguments for evaluating and evaluate the model"""
    parser = ArgumentParser()
    parser.add_argument("--config", help="Path to config file", type=str, required=False)
    parser.add_argument("--target-root", help="Path to output root", type=str, required=True)
    parser.add_argument("--model-name", help="Model name to evaluate", type=str, required=True)
    parser.add_argument(
        "--data-set",
        help="Data set to use",
        choices=["train", "test", "validate"],
        type=str,
        required=True,
    )
    epoch_args = parser.add_mutually_exclusive_group()
    epoch_args.add_argument(
        "--epoch", help="Evaluate this epoch", type=int, required=False, default=None
    )
    epoch_args.add_argument(
        "--min-epoch",
        help="Evaluate epochs starting from this",
        type=int,
        required=False,
        default=None,
    )
    epoch_args.add_argument(
        "--best-epoch",
        help="Use best epoch read from best_epoch.txt",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--batch-size", help="Batch size used in evaluating", type=int, required=False, default=None
    )
    parser.add_argument("--devices", help="Names of the devices to use", type=str, nargs="+")
    parser.add_argument(
        "--full-patch",
        help="Use only patches fully inside input valid region",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--plot-outputs", help="Plot outputs", default=False, action="store_true", required=False
    )
    parser.add_argument(
        "--shuffle",
        help="Shuffle evaluation order",
        default=False,
        action="store_true",
        required=False
    )
    parser.add_argument(
        "--sequence-item", help="Sequence item to evaluate", type=int, required=False, default=None
    )
    parser.add_argument(
        "--seed", help="Seed to use for generating the dataset", type=int, required=True
    )
    parser.add_argument(
        "--do-not-skip-evaluated",
        help="Do not skip evaluated",
        default=False,
        action="store_true",
        required=False,
    )
    args = parser.parse_args()
    target_dir = join(args.target_root, args.model_name)
    if args.config is None:
        config_path = join(target_dir, "training_config.json")
    else:
        config_path = args.config
    with open(config_path, mode="r", encoding="UTF-8") as config_file:
        config = json_load(config_file)
    if args.epoch is None:
        if args.best_epoch:
            with open(
                join(target_dir, "best_epoch.txt"), mode="r", encoding="UTF-8"
            ) as best_epoch_file:
                epochs: Iterable[int] = [int(best_epoch_file.read())]
        else:
            if args.min_epoch is None:
                min_epoch = 1
            else:
                min_epoch = args.min_epoch
            largest_epoch = find_largest_epoch(target_dir, "training")
            if largest_epoch is None:
                raise ValueError("Largest epoch not found!")
            epochs = reversed(range(min_epoch, largest_epoch + 1))
    else:
        epochs = [args.epoch]
    if args.batch_size is not None:
        config["data_loader"]["inference_batch_size"] = args.batch_size
    devices = [device(device_name) for device_name in args.devices]
    with no_grad():
        evaluating_function(
            config,
            target_dir,
            args.seed,
            epochs,
            args.plot_outputs,
            args.sequence_item,
            args.shuffle,
            args.data_set,
            args.full_patch,
            not args.do_not_skip_evaluated,
            devices,
        )
