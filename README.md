# cOA classification

Train and use neural networks to differentiate stoichiometries of cyclic oligo-adenylates (cOAs) in nanaopore signal.

## Setup
Install the provided conda environment, preferrably with mamba drop-in. From the main directory:
```shell
mamba env create -f environment.yaml
```

## Usage
Activate the environment before each usage:
```shell
conda activate coa_classifier
```

### Data preparation
Event data should be provided as directories of txt files with extension`.dat`, one directory per trace. Directory name should contain the cOA identifier and the trace identifier, the latter provided as `MXXX` (e.g. `cOA3_M011`, in full). Directories should be converted to npz files like so:

```shell
python coa_classifier/resources/txt2npz.py \
    --in-path path/to/dirs/ \
    --out-dir path/to/output_dir/
```

### Training
Cross-validated classifier generation and training is done in a single command:
```shell
python coa_classifier run_evaluation \
    --abf-in path/to/npz_dir/ \
    --out-dir path/to/output/
    
```

This will train a CNN on non-overlapping folds and test it on the held-out fold. The model is also trained once on the entire dataset for subsequent usage. All models are stored in the output directory, under  Note that each trace can only provide events to a single-fold, thus avoiding the risk of inflating accuracy measures due to learning of trace-specific features.

Some extra options are available:
- `--parameter-file`: choose a different neural network architecture by providing a yaml file. available yaml files are in `nns/hyperparams`. By default `CnnParameterFile_explicitLength_coa.yaml` is used.
- `--event-types`: choose which cOA stoichiometry classes are discerned by providing a yaml file. Available yaml files are in `coa_types`. By default `coa_types_3.yaml` is used, i.e. cA5, cA6 and a joined cA3/4 class.
- `--nb-folds`: Define number of folds to define in cross-validation scheme (default: 10)

### Inference
To run inference on a new sample and determine ratios of cOAs present:

```shell
python coa_classifier run_inference_bootstrapped \
    --nn-path path/to/your/nn.h5 \
    --abf-in path/to/your/npzs/ \
    --out-dir path/to/output/
```
This will also bootstrap your dataset 100 times to generate prediction intervals. 

Extra options:
- `--error-correct-rates`: Correct estimated relative concentrations by taking into account the different rates at which cOAs pass the nanpore. These rates were determined in monodisperse samples and are stored in  `resources/coa_rates.yaml`
- `--bootstrap-iters`: Adapt the number of bootstrap iterations (default: 100)

Alternatively, disperse your bootstrap iterations evenly over several models, e.g. the ones you got from the cross-validation procedure, to account for variation due to training procedure, as follows:

```shell
python coa_classifier run_inference_cv \
    --nn-dir path/to/dir/containing/nns/ \
    --abf-in path/to/your/npzs/ \
    --out-dir path/to/output/
  
```

## Reproducing paper figures

Traces and extracted events (`txt` and `npz`) can be obtained (here)[url_coming]. Trained models used in analysis can be found in this repo, under `nns/models`.

### Training
Starting from `.npz` files:
```shell
python coa_classifier run_evaluation \
    --abf-in npz_traces/monodisperse_npz/ \
    --out-dir path/to/output/
```

### Polydisperse samples with known composition
Either take the networks trained on CV folds or those in this repository (`nns/models/nn_expLen_oldNorm_3class_cv/`) and run inference on each of the polydisperse samples:
```shell
python coa_classifier run_inference_cv \
    --nn-dir path/to/dir/containing/nns/ \
    --abf-in npz_traces/mixes_npz/10_40_40_10 \
    --bootstrap-iters 1 \
    --out-dir path/to/output/
```

Barplots can then be generated as follows:
```shell
python coa_classifier/resources/mkae_mixture_barplot.py \
    --npy-in path/to/confmats_normalized.npy \
    --class-names cA3/4 cA5 cA6 \
    --ref-values 0.5 0.4 0.1   # Adapt these to the specific mixture!
```
The `confmats_normalized.npy` file can be found in the output directory generated in the previous step.

Repeat in similar fashion for the other mixtures.

### Samples of unknown composition
Run bootstrapped analysis on the typeIIIA and B samples. Use either the network provided in this repo (`nns/models/nn_expLen_oldNorm_3class.h5`) or the one from the first step:
```shell
python coa_classifier run_inference_bootstrapped \
    --nn-path path/to/your/nn.h5 \
    --abf-in npz_traces/typeIII_npz/typeIIIA/ \
    --out-dir path/to/output/
```

Repeat for typeIIIB in similar fashion.

## Cite this work
TBA