# Deformation equivariant cross-modality image synthesis with paired non-aligned training data

Official implementations for paper "Deformation equivariant cross-modality image synthesis with paired non-aligned training data" (Honkamaa et al., 2022)

## Environment setup

First install conda (https://docs.conda.io/en/latest/). To setup the enviroment navigate to directory ''devenv'' and execute ''setup.py'':

    python setup.py

The setup script will create a virtual enviroment with required packages installed. To activate the environent navigate to root and run:

    conda activate ./.venv


## Usage

For the remaining part we assume that you have navigated to directory ''src''. 

### Dataset generation

#### Cross-modality MRI synthesis dataset

To download and preprocess semi-synthetic cross-modality MRI synthesis dataset with simulated deformations, run the command (this can take a while):

    python -m scripts.generate_IXI_dataset --seed <seed_for_deformations> --robex-binary-path <path_to_ROBEX_binary> --elastix-binary-path <path_to_elastix_5.0.1_binary --transformix-binary-path <path_to_transformix_5.0.1_binary> --elastix-threads <number_of_threads_for_elastix>

ROBEX (Iglesias et al., 2011) binary can be downloaded from https://www.nitrc.org/projects/robex.

Elastix and transformix binaries are part of elastix registration software (Klein et al., 2010; Shamonin et al., 2014) and are used to re-register the synthetically deformed images images. Note that for linux and Mac path to the elastix libraries must be included in $LD_LIBRARY_PATH environment variable. Elastix 5.0.1 binaries can be downloaded from https://github.com/SuperElastix/elastix/releases/tag/5.0.1.

This data set uses publically available images from IXI data set http://brain-development.org/ixi-dataset/.

The most time consuming step in preprocessing is the re-registration step which is not required for our algorithm.

#### Synthetic data set

To download and preprocess synthetic data set based on COCO images, run the command (this can take a while):

    python -m scripts.generate_synthetic_dataset

This data set uses publically available COCO images from https://cocodataset.org/ (Lin et al., 2014).

#### Virtual staining data set

Our preprocessing applied to the virtual staining data set can be executed by (this will take a while)

    python -m scripts.preprocess_virtual_staining_dataset --source <path_to_the_downloaded_data_archive>

This data set uses publically available histopathology images by Latonen et al. (2022) and the source parameter in the command should refer to the "virtual-staining-data.tar" file which can be downloaded from https://doi.org/10.23729/9ddc2fc5-9bdb-404c-be07-c9c9540a32de.

#### Head MRI to CT synthesis data set

Our preprocessing applied to the head MRI to CT data set (based on CERMEP-iDB-MRXFDG database) can be executed by (this will take a while)

    python -m scripts.preprocess_cermep_dataset --source <path_to_the_obtained_data_archive> --robex-binary-path <path_to_ROBEX_binary> --elastix-binary-path <path_to_elastix_5.0.1_binary> --elastix-threads <number_of_threads_for_elastix>

The source parameter should refer to "iDB-CERMEP-MRXFDG_MRI_ct.tar.gz" archive which can be requested from the authors of CERMEP-iDB-MRXFDG database (Mérida, Inés, et al., 2021).

ROBEX (Iglesias et al., 2011) binary can be downloaded from https://www.nitrc.org/projects/robex.

Elastix binary is part of elastix registration software (Klein et al., 2010; Shamonin et al., 2014) and is used to re-register the synthetically deformed images images. Note that for linux and Mac path to the elastix libraries must be included in $LD_LIBRARY_PATH environment variable. Elastix 5.0.1 binaries can be downloaded from https://github.com/SuperElastix/elastix/releases/tag/5.0.1.

The most time consuming step in preprocessing is the registration step which is not required for our algorithm.

### Training

Two training scripts for our model are available:

 - scripts/training/train_non_aligned_i2i.py: Trains a model for cross-modality image synthesis with misaligned data
 - scripts/training/train_non_aligned_i2i_with_discriminator.py: Trains a model for cross-modality image synthesis with misaligned data, together with conditional adversarial loss.

Additionally scripts for baseline models are availabled at scripts/training/baseline.

Run the scripts as modules, e.g.

    python -m scripts.training.train_non_aligned_i2i <parameters>

The scripts have the following (main) parameters:

| Parameter | Description |
| --- | --- |
| --config | Path to config file, config files can be found in the directory "scripts/configs" |
| --target-root | Path to where trained models are saved |
| --model-name | Name of the trained model for saving |
| --continue-from-epoch | If specified, the training is continued from epoch after this |
| --devices | Devices to use, e.g. cuda:0 |
| --seed | Seed for data generation |

If you use configs from the "scripts/configs" directory, please use the correct script indicated by the names of the parent folders.

### Inference

Two inference scripts are available:

 - inference/inference_nifti.py: Combines a nifti volume from predicted image patches
 - inference/inference_tiff.py: Combines a tiff image from predicted image patches

Run the scripts as modules, e.g.

    python -m scripts.inference.inference_nifti <parameters>

 The scripts have the following (main) parameters:

| Parameter | Description |
| --- | --- |
| --target-root | Path to where trained models are saved |
| --model-name | Name of the trained model |
| --data-set | Which data set split to use, either "train", "test", or "validate" |
| --epoch | Which epoch to use for inference |
| --devices | Devices to use |
| --save-slices-over-dims | (Optional) Save predicted 2D slices over selected dimensions. Can be used for computing the FID metrics (explained below). |

The inference results will be stored in the target root at the models directory.

### Evaluation

Evaluation scripts are available at scripts/evaluation. The scripts at the subfolder scripts/evaluation/model_output directly evaluate models as they are generating the outputs whereas the scripts at scripts/evaluation/full_volume evaluate full volumes after running the inference.

The exact commands for each data set are provided below. In all of the cases the data set paramater can be either "validate" or "test".

FID-score evaluation can be done separately using the --save-slices-over-dims flag for the inference scripts described above, and the PyTorch FID-score implementation https://github.com/mseitzer/pytorch-fid (Seitzer, 2020).

Evaluation results will be stored in the target root at the models directory.

#### Cross-modality brain MRI synthesis dataset

For evaluating cross-modality brain MRI synthesis models, one can use the following command

    python -m scripts.evaluation.full_volume.evaluate_i2i --target-root <target_root_path> --model-name <trained_model_name> --data-set <data_set> --epoch <epoch_to_evaluate>

Before running the evaluation, inference must have been executed for given data set and epoch.

#### Synthetic data set

For evaluating synthetic COCO data set models, one can use the following command

    python -m scripts.evaluation.model_output.evaluate_i2i --target-root <target_root_path> --model-name <trained_model_name> --data-set <data_set> --epoch <epoch_to_evaluate> --devices <devices> --seed <seed_for_test_dataset_generation>

#### Virtual staining data set

For evaluating virtual staining models (only pixel metrics), one can use the following command

    python -m scripts.evaluation.model_output.evaluate_i2i --target-root <target_root_path> --model-name <trained_model_name> --data-set <data_set> --epoch <epoch_to_evaluate> --devices <devices> --seed <seed_for_test_dataset_generation> --full-patch

The nuclei reproducibility evaluation is not included in this codebase but the trained detection model is public and provided as a supplementary material for the paper by Valkonen et al. (2020), and can be found at https://github.com/BioimageInformaticsTampere/NucleiDetection.

#### MRI to CT synthesis data set

For evaluating head MRI to CT synthesis models, one can use the following command

    python -m scripts.evaluation.full_volume.evaluate_CERMEP --target-root <target_root_path> --model-name <trained_model_name> --data-set <data_set> --epoch <epoch_to_evaluate> --n-evaluation-locations-per-volume 20 --registration-mask-width 100.0 --evaluation-mask-width 30.0 --n-evaluation-processes <number_of_evaluation_processes> --seed <seed_for_registration> -transformations-cache-path <cache_path_for_transformation> --elastix-binary-path <path_to_elastix_5.0.1_binary> --elastix-threads <number_of_threads_for_elastix>

Before running the evaluation, inference must have been executed for given data set and epoch.

The parameter "cache_path_for_transformation" can be optionally given to cache transformations to given directory to avoid recomputing registrations every time (which is very time consuming). It is recommended to use large number of evaluation processes.

Elastix binary is part of elastix registration software (Klein et al., 2010; Shamonin et al., 2014) and is used to re-register the synthetically deformed images images. Note that for linux and Mac path to the elastix libraries must be included in $LD_LIBRARY_PATH environment variable. Elastix 5.0.1 binaries can be downloaded from https://github.com/SuperElastix/elastix/releases/tag/5.0.1.

For computing the normalized mutual information for the MRI to CT synthesis data set we used the same command used for evaluating cross-modality brain MRI synthesis dataset.

## Publication

If you use the repository, please cite (see [bibtex](citations.bib)):

- **Deformation equivariant cross-modality image synthesis with paired non-aligned training data**  
[Joel Honkamaa](https://github.com/honkamj "Joel Honkamaa"), Umair Khan, Sonja Koivukoski, Mira Valkonen, Leena Latonen, Pekka Ruusuvuori, and Pekka Marttinen  
Accepted for publication ([eprint arXiv:2208.12491](https://arxiv.org/abs/2208.12491 "eprint arXiv:2208.12491"))

## License

The codebase is released under the MIT license.

## Bibliography

 - Iglesias, Juan Eugenio, et al. "Robust brain extraction across datasets and comparison with publicly available methods." IEEE transactions on medical imaging 30.9 (2011): 1617-1634.
 -  Klein, Stefan, et al. "Elastix: a toolbox for intensity-based medical image registration." IEEE transactions on medical imaging 29.1 (2009): 196-205.
 - Lin, Tsung-Yi, et al. "Microsoft coco: Common objects in context." Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13. Springer International Publishing, 2014.
 - Mérida, Inés, et al. "CERMEP-IDB-MRXFDG: a database of 37 normal adult human brain [18F] FDG PET, T1 and FLAIR MRI, and CT images available for research." EJNMMI research 11.1 (2021): 1-10.
 - Seitzer, Maximilian. "pytorch-fid: FID Score for PyTorch." (2020).
 - Shamonin, Denis P., et al. "Fast parallel image registration on CPU and GPU for diagnostic classification of Alzheimer's disease." Frontiers in neuroinformatics 7 (2014): 50.
 - Valkonen, Mira, et al. "Generalized fixation invariant nuclei detection through domain adaptation based deep learning." IEEE Journal of Biomedical and Health Informatics 25.5 (2020): 1747-1757.