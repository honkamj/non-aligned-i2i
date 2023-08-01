"""Evaluation script for full volume evaluation with ground truth"""


from argparse import ArgumentParser
from json import load as json_load
from os import listdir, makedirs
from os.path import join
from typing import Any, Iterable, Mapping, NamedTuple

from nibabel import load as nib_load  # type: ignore
from numpy import abs as np_abs
from numpy import ndarray
from tqdm import tqdm  # type: ignore

from data.util import get_data_root
from metrics.normalized_mutual_information import normalized_mutual_information
from metrics.peak_signal_to_noise_ratio import PSNRMassFunction
from metrics.structural_similarity_index import structural_similarity_index
from util.training import LossAverager, find_largest_epoch


class _EvaluationArgs(NamedTuple):
    """Evaluation arguments"""

    config: Mapping[str, Any]
    target_dir: str
    epoch: int
    data_set: str


def _load_images(
    args: _EvaluationArgs, case_name: str
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    data_folder_path = join(get_data_root(args.config["data"]["root"]), args.data_set)
    inference_folder_path = join(
        args.target_dir, "inference", f"epoch{int(args.epoch):03d}", args.data_set
    )
    input_path = join(
        data_folder_path,
        case_name,
        f"{case_name}_{args.config['data']['postfixes']['input']}",
    )
    input_mask_path = join(
        data_folder_path,
        case_name,
        f"{case_name}_{args.config['data']['postfixes']['input_mask']}",
    )
    prediction_path = join(inference_folder_path, case_name, f"{case_name}_predicted.nii.gz")
    label_aligned_path = join(
        data_folder_path,
        case_name,
        f"{case_name}_{args.config['data']['postfixes']['label_aligned']}",
    )
    label_aligned_mask_path = join(
        data_folder_path,
        case_name,
        (f"{case_name}_" f"{args.config['data']['postfixes']['label_aligned_mask']}"),
    )
    evaluation_mask_path = join(
        data_folder_path,
        case_name,
        f"{case_name}_{args.config['data']['postfixes']['evaluation_mask']}",
    )

    return (
        nib_load(input_path).get_fdata(),
        nib_load(input_mask_path).get_fdata(),
        nib_load(prediction_path).get_fdata(),
        nib_load(label_aligned_path).get_fdata(),
        nib_load(label_aligned_mask_path).get_fdata(),
        nib_load(evaluation_mask_path).get_fdata(),
    )


def _evaluate(args: _EvaluationArgs):
    metric_averager = LossAverager()
    metric_averager.set_custom_mass_func(
        "PSNR", PSNRMassFunction(args.config["data_loader"]["label_signal_range"])
    )
    data_folder_path = join(get_data_root(args.config["data"]["root"]), args.data_set)
    case_metrics_target_dir = join(
        args.target_dir, f"{args.data_set}_full_volume_evaluation_metrics"
    )
    makedirs(case_metrics_target_dir, exist_ok=True)
    case_names = listdir(data_folder_path)
    evaluation_tqdm = tqdm(sorted(case_names))
    for case_name in evaluation_tqdm:
        (
            input_volume,
            input_mask,
            prediction,
            label,
            label_mask,
            evaluation_mask,
        ) = _load_images(args, case_name)
        combined_mask = evaluation_mask * input_mask * label_mask
        mask_mass = combined_mask.sum()
        squared_sum = ((label - prediction) ** 2 * combined_mask).sum()
        absolute_sum = (np_abs(label - prediction) * combined_mask).sum()
        (
            structural_similarity_sum,
            structural_similarity_averaging_mass,
        ) = structural_similarity_index(
            label=label[None, None],
            output=prediction[None, None],
            content_mask=(input_mask * label_mask)[None, None],
            evaluation_mask=evaluation_mask[None, None],
            data_range=args.config["data_loader"]["label_signal_range"],
        )
        (
            nmi_sum,
            nmi_averaging_mass
        ) = normalized_mutual_information(
            label=input_volume[None, None],
            output=prediction[None, None],
            mask=combined_mask[None, None]
        )
        case_metric_averager = LossAverager()
        case_metric_averager.set_custom_mass_func(
            "PSNR", PSNRMassFunction(args.config["data_loader"]["label_signal_range"])
        )
        metric_averagers = [metric_averager, case_metric_averager]
        for single_metric_averager in metric_averagers:
            single_metric_averager.count({"MSE": squared_sum}, mass=mask_mass)
            single_metric_averager.count({"MAE": absolute_sum}, mass=mask_mass)
            single_metric_averager.count({"PSNR": squared_sum}, mass=mask_mass)
            single_metric_averager.count(
                {"SSIM": structural_similarity_sum}, mass=structural_similarity_averaging_mass
            )
            single_metric_averager.count(
                {"NMI": nmi_sum}, mass=nmi_averaging_mass
            )
        evaluation_tqdm.set_description(str(metric_averager))
        case_metric_averager.save_to_json(
            epoch=args.epoch,
            filename=join(case_metrics_target_dir, f"{case_name}.json"),
            postfix="",
        )
    metric_averager.save_to_json(
        epoch=args.epoch,
        filename=join(args.target_dir, f"{args.data_set}_full_volume_evaluation_metrics.json"),
        postfix="",
    )


def _main() -> None:
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
    for epoch in epochs:
        _evaluate(
            _EvaluationArgs(
                config=config,
                target_dir=target_dir,
                epoch=epoch,
                data_set=args.data_set,
            )
        )


if __name__ == "__main__":
    _main()
