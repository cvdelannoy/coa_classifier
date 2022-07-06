import os

import yaml
from jinja2 import Template
from snakemake import snakemake

from resources.helper_functions import parse_output_path

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def main(args):
    db_dir = parse_output_path(f'{args.out_dir}dbs/')
    nn_dir = parse_output_path(f'{args.out_dir}nns/')
    logs_dir = parse_output_path(f'{args.out_dir}logs/')
    with open(args.parameter_file, 'r') as pf: params = yaml.load(pf, Loader=yaml.FullLoader)

    # Construct and run snakemake pipeline
    with open(f'{__location__}/run_production_pipeline.sf', 'r') as fh: template_txt = fh.read()
    sm_text = Template(template_txt).render(
        __location__=__location__,
        db_dir=db_dir,
        nn_dir=nn_dir,
        logs_dir=logs_dir,
        parameter_file=args.parameter_file,
        train_reads=args.training_abfs,
        test_reads=args.test_abfs,
        filter_width=params['filter_width'],
    )

    sf_fn = f'{args.out_dir}nn_production_pipeline.sf'
    with open(sf_fn, 'w') as fh: fh.write(sm_text)
    snakemake(sf_fn, cores=args.cores, verbose=False, keepgoing=True, resources={'gpu': 1})
