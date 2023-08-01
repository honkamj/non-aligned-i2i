"""Simple inference script for tiff-data"""

from argparse import ArgumentParser
from contextlib import ExitStack
from json import load as json_load
from os import makedirs
from os.path import join
from shutil import make_archive, rmtree
from tempfile import TemporaryDirectory
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

from matplotlib.image import imsave  # type: ignore
from numpy import asarray, clip, divide
from numpy import dtype as np_dtype
from numpy import ndarray, zeros, zeros_like
from tifffile import TiffFile, imwrite  # type: ignore
from torch import device, no_grad
from torch import sum as torch_sum
from tqdm import tqdm  # type: ignore

from data.interface import init_generic_inference_data_loader
from data.patch_sequence import Patch
from data.util import get_data_root
from model.inference import get_generic_i2i_inference_function
from util.training import find_largest_epoch, get_path


def _get_samples(
    data_config: Mapping[str, Any], data_set: str
) -> Sequence[Tuple[str, Tuple[int, ...], np_dtype, str]]:
    sample_information: List[Tuple[str, Tuple[int, ...], np_dtype, str]] = []
    for inference_sample in sorted(data_config["datasets"][data_set]):
        file_path = get_path(
            data_root=get_data_root(data_config["root"]),
            postfix=data_config["postfixes"]["label_training"],
            sample=inference_sample,
        )
        with TiffFile(file_path, mode="rb") as tiff_file:
            shape = tiff_file.pages[0].shape
            dtype = tiff_file.pages[0].dtype
            photometric = tiff_file.pages[0].photometric
        sample_information.append((inference_sample, shape, dtype, photometric))
    return sample_information


def _denormalize_output(output: ndarray, normalization_config: Mapping[str, Any]) -> ndarray:
    mean_and_std = asarray(normalization_config["label_mean_and_std"])
    output = output * mean_and_std[1] + mean_and_std[0]
    output = clip(output, a_min=0, a_max=255)
    output = output.round()
    return output


def _inference(
    config: Mapping[str, Any],
    target_dir: str,
    epoch: int,
    data_set: str,
    devices: Sequence[device],
    save_full_tiles: bool,
) -> None:
    torch_device = devices[0]
    inference_function = get_generic_i2i_inference_function(
        model_config=config["model"],
        data_loader_config=config["data_loader"],
        devices=devices,
        epoch=epoch,
        target_dir=target_dir,
    )
    sample_tqdm = tqdm(_get_samples(config["data"], data_set), unit="sample")
    with ExitStack() as exit_stack:
        if save_full_tiles:
            tile_index = 0
            tile_dir = exit_stack.enter_context(TemporaryDirectory())
        for sample, sample_shape, sample_dtype, sample_photometric in sample_tqdm:
            sample_tqdm.write(f"Inference for {sample}")
            data_loader = init_generic_inference_data_loader(
                sample=sample, data_config=config["data"], data_loader_config=config["data_loader"]
            )

            target_array = zeros(sample_shape, dtype="float32")
            fusing_mask_array = zeros(tuple(sample_shape[:-1]) + (1,), dtype="float32")
            data_tqdm = tqdm(data_loader, leave=False)
            for (input_image, input_mask, fusing_mask, patch_export) in data_tqdm:
                input_image = input_image.to(torch_device)
                input_mask = input_mask.to(torch_device)
                fusing_mask = fusing_mask.to(torch_device)
                if torch_sum(input_mask * fusing_mask) < 1e-5:
                    continue
                predicted_label = inference_function(input_image)
                predicted_label = predicted_label * input_mask
                for (
                    single_patch_export,
                    single_predicted_label,
                    single_fusing_mask,
                    single_input_mask,
                ) in zip(patch_export, predicted_label, fusing_mask, input_mask):
                    patch = Patch.from_numpy(single_patch_export.numpy())
                    target_array[patch.get_slice()] += (
                        (single_predicted_label * single_fusing_mask)
                        .cpu()
                        .numpy()
                        .transpose(1, 2, 0)
                    )
                    fusing_mask_array[patch.get_slice()] += (
                        single_fusing_mask.cpu().numpy().transpose(1, 2, 0)
                    )
                    if save_full_tiles:
                        if single_input_mask.mean() == 1.0:
                            normalized_tile = _denormalize_output(
                                single_predicted_label.cpu().numpy().transpose(1, 2, 0),
                                config["data_loader"]["normalization"],
                            ).astype(sample_dtype)
                            imsave(
                                join(tile_dir, f"{tile_index}.png"),
                                normalized_tile,
                                vmin=0,
                                vmax=255,
                            )
                            tile_index += 1
            target_array = divide(
                target_array,
                fusing_mask_array,
                out=zeros_like(target_array),
                where=fusing_mask_array != 0,
            )
            del fusing_mask_array
            target_array = _denormalize_output(
                target_array, config["data_loader"]["normalization"]
            ).astype(sample_dtype)

            target_folder = join(
                target_dir, "inference", f"epoch{int(epoch):03d}", data_set, sample
            )
            makedirs(target_folder, exist_ok=True)
            imwrite(
                join(target_folder, f"{sample}_stained_predicted.tif"),
                target_array,
                photometric=sample_photometric,
            )
        if save_full_tiles:
            output_tile_zip = join(
                target_dir, "inference", f"epoch{int(epoch):03d}", data_set, f"tiles_{data_set}"
            )
            make_archive(output_tile_zip, "zip", tile_dir)
            rmtree(tile_dir)


def _main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--target-root", help="Path to output root", type=str, required=True)
    parser.add_argument(
        "--data-set",
        help="Data set to use",
        choices=["train", "test", "validate"],
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model-name", help="Model name used in saving the model", type=str, required=True
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
        default=None,
        action="store_true",
        required=False,
    )
    parser.add_argument("--devices", help="Names of the devices to use", type=str, nargs="+")
    parser.add_argument(
        "--save-full-tiles",
        help="Save tiles fully within mask to inference folder",
        default=False,
        action="store_true",
        required=False,
    )
    args = parser.parse_args()
    config_path = join(args.target_root, args.model_name, "training_config.json")
    with open(config_path, mode="r", encoding="UTF-8") as config_file:
        config = json_load(config_file)
    target_dir = join(args.target_root, args.model_name)
    devices = [device(device_name) for device_name in args.devices]
    if args.epoch is None:
        if args.best_epoch is not None:
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
    for epoch in epochs:
        with no_grad():
            _inference(config, target_dir, epoch, args.data_set, devices, args.save_full_tiles)


if __name__ == "__main__":
    _main()
