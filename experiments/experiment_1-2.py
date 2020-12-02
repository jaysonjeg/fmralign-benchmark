import itertools
import warnings
from joblib import Parallel, delayed
from fmralignbench.utils import (
    WHOLEBRAIN_DATASETS, ROI_DATASETS, inter_subject_align_decode, within_subject_decoding)
from fmralignbench.conf import ROOT_FOLDER, N_JOBS
from fmralignbench.plot_utils import make_bench_figure, make_within_subject_decoding_figure
warnings.filterwarnings(action='once')

input_methods = ["anat_inter_subject", "pairwise_scaled_orthogonal",
                 "pairwise_ot_e-1",  "srm", "intra_subject", "HA"]


if N_JOBS / 10 > 1:
    n_pipes = int(N_JOBS / 10)
    n_jobs = N_JOBS % 10
###### EXPERIMENT 1 #######

# Inter-subject results
experiment_parameters = list(itertools.product(
    WHOLEBRAIN_DATASETS, input_methods))

Parallel(n_jobs=n_pipes)(delayed(inter_subject_align_decode)(input_method, dataset_params, "schaefer",
                                                             ROOT_FOLDER, n_pieces=300, n_jobs=n_jobs) for dataset_params, input_method in experiment_parameters)

ROI = False
make_bench_figure(ROI)

# Within-subject results

Parallel(n_jobs=n_pipes)(delayed(within_subject_decoding)(dataset_params, ROOT_FOLDER)
                         for dataset_params in WHOLEBRAIN_DATASETS)
make_within_subject_decoding_figure()

###### EXPERIMENT 2 #######

experiment_parameters = list(itertools.product(ROI_DATASETS, input_methods))
Parallel(n_jobs=n_pipes)(delayed(inter_subject_align_decode)(input_method, dataset_params, "schaefer",
                                                             ROOT_FOLDER, n_pieces=300, n_jobs=n_jobs) for dataset_params, input_method in experiment_parameters)

ROI = True
make_bench_figure(ROI)
