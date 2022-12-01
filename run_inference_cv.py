import os
from run_inference import run_inference
from run_inference_bootstrap import run_inference_bootstrapped
from resources.run_inference_performance_analysis import run_inference_performance_analysis
from resources.helper_functions import parse_output_path, parse_input_path

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def main(args):
    nn_list = parse_input_path(args.nn_dir, pattern='*.h5')
    nn_list.sort()
    bootstrap_iters_per_model = args.bootstrap_iters // len(nn_list)
    out_dir = parse_output_path(args.out_dir, clean=True)
    inference_dir = parse_output_path(f'{out_dir}inference')
    for nni, nn in enumerate(nn_list):
        inf_dir = parse_output_path(f'{inference_dir}{nni}')
        run_inference_bootstrapped(args.abf_in, nn, inf_dir,
                                   bootstrap_iters_per_model, False, False, args.cores, False)
    run_inference_performance_analysis(inference_dir, f'{out_dir}analysis', True, False)