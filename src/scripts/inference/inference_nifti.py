"""Simple inference script for nifti-data"""

from argparse import ArgumentParser
from contextlib import ExitStack
from distutils.archive_util import make_archive
from json import load as json_load
from os import makedirs
from os.path import join
from tempfile import TemporaryDirectory
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from matplotlib.image import imsave  # type: ignore
from nibabel import Nifti1Image  # type: ignore
from nibabel import load as nib_load
from nibabel import save as nib_save
from numpy import clip, divide
from numpy import dtype as np_dtype
from numpy import moveaxis, ndarray
from numpy import round as np_round
from numpy import zeros, zeros_like
from skimage.color import gray2rgb  # type: ignore
from torch import Tensor, device, no_grad
from torch import sum as torch_sum
from torch import tensor
from tqdm import tqdm  # type: ignore

from data.interface import init_generic_inference_data_loader
from data.nifti.interface import get_nifti_samples
from data.patch_sequence import Patch
from data.util import get_data_root
from model.inference import get_generic_i2i_inference_function
from util.data import denormalize_tensor as denormalize
from util.training import find_largest_epoch, get_path


def _get_samples(
    data_config: Mapping[str, Any], data_set: str
) -> Sequence[Tuple[str, Tuple[int, ...], np_dtype, ndarray]]:
    root_directory = get_data_root(data_config["root"])
    samples = get_nifti_samples(root_directory, data_set)
    sample_information: List[Tuple[str, Tuple[int, ...], np_dtype, ndarray]] = []
    for inference_sample in sorted(samples):
        file_path = get_path(
            data_root=join(root_directory, data_set),
            postfix=data_config["postfixes"]["input"],
            sample=inference_sample,
        )
        nib_file = nib_load(file_path)
        sample_information.append(
            (inference_sample, nib_file.shape, nib_file.dataobj.dtype, nib_file.affine)
        )
    return sample_information


def _denormalize_output(output: Tensor, normalization_config: Mapping[str, Any]) -> Tensor:
    mean_and_std = tensor(normalization_config["label_mean_and_std"], device=output.device)
    denormalized = denormalize(output, mean_and_std)
    return denormalized


def _inference(
    config: Mapping[str, Any],
    target_dir: str,
    epoch: int,
    case: Optional[str],
    data_set: str,
    devices: Sequence[device],
    save_slices_over_dims: Optional[Sequence[int]],
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
        if save_slices_over_dims is not None:
            slice_indices = {}
            slice_dirs = {}
            temporary_directory = exit_stack.enter_context(TemporaryDirectory())
            for dim in save_slices_over_dims:
                slice_indices[dim] = 0
                slice_dirs[dim] = join(
                    temporary_directory,
                    f"slices_dim_{dim}",
                )
                makedirs(slice_dirs[dim])
        for sample, sample_shape, sample_dtype, sample_affine in sample_tqdm:
            if case is not None and sample != case:
                continue
            target_folder = join(
                target_dir, "inference", f"epoch{int(epoch):03d}", data_set, sample
            )
            target_path = join(target_folder, f"{sample}_predicted.nii.gz")
            sample_tqdm.write(f"Inference for {sample}")
            data_loader = init_generic_inference_data_loader(
                sample=join(data_set, sample),
                data_config=config["data"],
                data_loader_config=config["data_loader"],
            )

            target_array = zeros(sample_shape, dtype=sample_dtype)
            fusing_mask_array = zeros(sample_shape, dtype="float32")
            data_tqdm = tqdm(data_loader, leave=False)
            for (input_image, input_mask, fusing_mask, patch_export) in data_tqdm:
                input_image = input_image.to(torch_device)
                input_mask = input_mask.to(torch_device)
                fusing_mask = fusing_mask.to(torch_device)
                if torch_sum(input_mask * fusing_mask) < 1:
                    continue
                predicted_label = inference_function(input_image)
                predicted_label = (
                    _denormalize_output(predicted_label, config["data_loader"]["normalization"])
                    * input_mask
                    * fusing_mask
                )
                for single_patch_export, single_predicted_label, single_fusing_mask in zip(
                    patch_export, predicted_label.cpu(), fusing_mask.cpu()
                ):
                    patch = Patch.from_numpy(single_patch_export.numpy())
                    single_predicted_label_numpy = single_predicted_label.detach().numpy()
                    single_predicted_label_numpy = single_predicted_label_numpy.astype(sample_dtype)
                    target_array[patch.get_slice()] += single_predicted_label_numpy[0]
                    fusing_mask_array[patch.get_slice()] += single_fusing_mask.detach().numpy()[0]
            target_array = divide(
                target_array,
                fusing_mask_array,
                out=zeros_like(target_array),
                where=fusing_mask_array != 0,
            )
            if save_slices_over_dims is not None:
                for dim in save_slices_over_dims:
                    permuted_target_array = moveaxis(target_array, dim, 0)
                    for image_slice in permuted_target_array:
                        image_normalized_slice = np_round(
                            clip(
                                255
                                * (image_slice - config["data_loader"]["label_background_value"])
                                / config["data_loader"]["label_signal_range"],
                                a_min=0,
                                a_max=255,
                            )
                        ).astype("uint8")
                        imsave(
                            join(slice_dirs[dim], f"{slice_indices[dim]}.png"),
                            gray2rgb(image_normalized_slice),
                            vmin=0,
                            vmax=255,
                        )
                        slice_indices[dim] += 1
            new_image = Nifti1Image(target_array, sample_affine)
            makedirs(target_folder, exist_ok=True)
            nib_save(new_image, target_path)
        if save_slices_over_dims is not None:
            for dim in save_slices_over_dims:
                output_slice_zip = join(
                    target_dir,
                    "inference",
                    f"epoch{int(epoch):03d}",
                    data_set,
                    f"tiles_{data_set}_dim_{dim}",
                )
                make_archive(output_slice_zip, "zip", slice_dirs[dim])


def _main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--target-root", help="Path to output root", type=str, required=True)
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
    parser.add_argument("--data-set", help="Dataset to infer", type=str, required=True)
    parser.add_argument("--only-case", help="Only for case", type=str, required=False)
    parser.add_argument("--devices", help="Names of the devices to use", type=str, nargs="+")
    parser.add_argument(
        "--save-slices-over-dims",
        help="Save slices over a specified dimension as png-images",
        type=int,
        nargs="*",
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
            _inference(
                config,
                target_dir,
                epoch,
                args.only_case,
                args.data_set,
                devices,
                args.save_slices_over_dims,
            )


if __name__ == "__main__":
    _main()
