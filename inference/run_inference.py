import sys, signal, os, h5py
from datetime import datetime
import tensorflow as tf
import numpy as np
from helper_functions import parse_output_path
from inference.ReadTable import ReadTable

def main(args):
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    print('Loading model...')
    mod = tf.keras.models.load_model(args.model)
    with h5py.File(args.model, 'r') as fh:
        kmer_list = fh.attrs['compound_list'].split(',')
    print(f'Done!')
    pos_reads_dir = parse_output_path(args.out_dir + 'pos_reads')

    abundance_array = np.zeros(len(kmer_list))

    # Load read table, start table manager
    read_table = ReadTable(args.abf_in, pos_reads_dir)
    read_manager_process = read_table.init_table()

    # ensure processes are ended after exit
    def graceful_shutdown(sig, frame):
        print("shutting down")
        read_manager_process.terminate()
        read_manager_process.join()
        sys.exit(0)

    # Differentiate between inference modes by defining when loop stops
    if args.inference_mode == 'watch':
        signal.signal(signal.SIGINT, graceful_shutdown)
        def end_condition():
            return True
    elif args.inference_mode == 'once':
        def end_condition():
            return len(os.listdir(args.abf_in)) != 0
    else:
        raise ValueError(f'{args.inference_mode} is not a valid inference mode')
    start_time = datetime.now()
    # Start inference loop
    while end_condition():
        read_id, read_list = read_table.get_read_to_predict()
        if read_list is None: continue
        for read in read_list:
            read_tensor = tf.expand_dims(tf.ragged.constant([read]), -1)
            abundance_array += mod(read_tensor).numpy()
        read_table.update_prediction(read_id, np.zeros(len(read_id), dtype=bool))
    else:
        run_time = datetime.now() - start_time
        read_manager_process.terminate()
        read_manager_process.join()
        print(f'rutime was {run_time.seconds} s')
        freq_array = abundance_array / max(abundance_array.sum(), 1)
        abundance_txt = 'kmer,abundance,frequency\n' + \
                        '\n'.join([f'{km},{ab},{fr}' for km, ab, fr in zip(kmer_list, abundance_array, freq_array)])
        with open(f'{args.out_dir}abundance_estimation.csv', 'w') as fh:
            fh.write(abundance_txt)

