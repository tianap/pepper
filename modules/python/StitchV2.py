import h5py
import sys
from os.path import isfile, join
from os import listdir
import concurrent.futures
import numpy as np
from collections import defaultdict
from modules.python.Options import CandidateOptions


BASE_ERROR_RATE = 0.0
label_decoder = {1: 'A', 2: 'C', 3: 'G', 4: 'T', 0: ''}
MATCH_PENALTY = 4
MISMATCH_PENALTY = 6
GAP_PENALTY = 8
GAP_EXTEND_PENALTY = 2
MIN_SEQUENCE_REQUIRED_FOR_MULTITHREADING = 2


def decode_ref_base(ref_code):
    if ref_code == 0:
        return 'N'
    elif ref_code == 1:
        return 'A'
    elif ref_code == 2:
        return 'C'
    elif ref_code == 3:
        return 'G'
    elif ref_code == 4:
        return 'T'


def decode_bases(pred_code):
    if pred_code == 0:
        return ['*', '*']
    elif pred_code == 1:
        return ['A', 'A']
    elif pred_code == 2:
        return ['A', 'C']
    elif pred_code == 3:
        return ['A', 'T']
    elif pred_code == 4:
        return ['A', 'G']
    elif pred_code == 5:
        return ['A', '*']
    elif pred_code == 6:
        return ['C', 'C']
    elif pred_code == 7:
        return ['C', 'T']
    elif pred_code == 8:
        return ['C', 'G']
    elif pred_code == 9:
        return ['C', '*']
    elif pred_code == 10:
        return ['T', 'T']
    elif pred_code == 11:
        return ['T', 'G']
    elif pred_code == 12:
        return ['T', '*']
    elif pred_code == 13:
        return ['G', 'G']
    elif pred_code == 14:
        return ['G', '*']


def get_file_paths_from_directory(directory_path):
    """
    Returns all paths of files given a directory path
    :param directory_path: Path to the directory
    :return: A list of paths of files
    """
    file_paths = [join(directory_path, file) for file in listdir(directory_path)
                  if isfile(join(directory_path, file)) and file[-2:] == 'h5']
    return file_paths


def chunks(file_names, threads):
    """Yield successive n-sized chunks from l."""
    chunks = []
    for i in range(0, len(file_names), threads):
        chunks.append(file_names[i:i + threads])
    return chunks


def chunks_alignment_sequence(alignment_sequence_pairs, min_length):
    """Yield successive n-sized chunks from l."""
    chunks = []
    for i in range(0, len(alignment_sequence_pairs), min_length):
        chunks.append(alignment_sequence_pairs[i:i + min_length])
    return chunks


def find_candidates(file_name, contig, small_chunk_keys):
    # for chunk_key in small_chunk_keys:
    candidate_variants = defaultdict(list)
    reference_dict = defaultdict()

    for contig_name, _st, _end in small_chunk_keys:
        chunk_name = contig_name + '-' + str(_st) + '-' + str(_end)

        with h5py.File(file_name, 'r') as hdf5_file:
            smaller_chunks = set(hdf5_file['predictions'][contig][chunk_name].keys()) - {'contig_start', 'contig_end'}

        smaller_chunks = sorted(smaller_chunks)
        base_prediction_dict = defaultdict()

        for chunk in smaller_chunks:
            with h5py.File(file_name, 'r') as hdf5_file:
                bases = hdf5_file['predictions'][contig][chunk_name][chunk]['bases'][()]
                positions = hdf5_file['predictions'][contig][chunk_name][chunk]['position'][()]
                indices = hdf5_file['predictions'][contig][chunk_name][chunk]['index'][()]
                ref_seq = hdf5_file['predictions'][contig][chunk_name][chunk]['ref_seq'][()]

            positions = np.array(positions, dtype=np.int64)
            base_predictions = np.array(bases, dtype=np.int)

            for pos, indx, base_pred, ref_code in zip(positions, indices, base_predictions, ref_seq):
                if indx < 0 or pos < 0 or indx > 0:
                    continue
                # update the reference base
                reference_base = decode_ref_base(ref_code)

                # get the predictions and see if this site has a candidate
                base_pred_norm = base_pred / sum(base_pred)

                # first see if this position has any candidates
                has_candidate = False
                for pred_code, pred_value in enumerate(base_pred_norm):
                    if pred_value >= CandidateOptions.CANDIDATE_PROB_THRESHOLD:
                        pred_bases = decode_bases(pred_code)
                        for pred_base in pred_bases:
                            if pred_base != reference_base:
                                has_candidate = True
                                break

                # if it has candidates, then update the dictionaries
                if has_candidate:
                    for pred_code, pred_value in enumerate(base_pred_norm):
                        if pred_value >= CandidateOptions.CANDIDATE_PROB_THRESHOLD:
                            pred_bases = decode_bases(pred_code)
                            predicted_bases = pred_bases

                            # candidate bases
                            reference_dict[pos] = reference_base
                            if predicted_bases not in candidate_variants[pos]:
                                candidate_variants[pos].append(predicted_bases)

    return contig, reference_dict, candidate_variants


def create_consensus_sequence(hdf5_file_path, contig, sequence_chunk_keys, threads):
    sequence_chunk_keys = sorted(sequence_chunk_keys)
    sequence_chunk_key_list = list()
    for sequence_chunk_key in sequence_chunk_keys:
        contig, st, end = sequence_chunk_key.split('-')
        sequence_chunk_key_list.append((contig, int(st), int(end)))

    sequence_chunk_key_list = sorted(sequence_chunk_key_list, key=lambda element: (element[1], element[2]))
    all_candidates = defaultdict(list)
    global_reference_dict = defaultdict()
    positions_with_candidates = list()

    # generate the dictionary in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        file_chunks = chunks(sequence_chunk_key_list, max(MIN_SEQUENCE_REQUIRED_FOR_MULTITHREADING,
                                                          int(len(sequence_chunk_key_list) / threads) + 1))

        futures = [executor.submit(find_candidates, hdf5_file_path, contig, file_chunk)
                   for file_chunk in file_chunks]
        for fut in concurrent.futures.as_completed(futures):
            if fut.exception() is None:
                contig, reference_dict, candidate_variants = fut.result()
                for pos, alt_alleles in candidate_variants.items():
                    if alt_alleles:
                        global_reference_dict[pos] = reference_dict[pos]
                        positions_with_candidates.append(pos)
                        all_candidates[pos].extend(alt_alleles)
            else:
                sys.stderr.write("ERROR: " + str(fut.exception()) + "\n")
            fut._result = None  # python issue 27144

    return all_candidates, global_reference_dict, positions_with_candidates
