import h5py
import argparse
import time
import sys
from os.path import isfile, join
from os import listdir
import concurrent.futures
import numpy as np
from collections import defaultdict
import operator
from modules.python.TextColor import TextColor
from build import PEPPER
import re


BASE_ERROR_RATE = 0.0
label_decoder = {1: 'A', 2: 'C', 3: 'G', 4: 'T', 0: ''}
label_decoder_snp = {1: 'A', 2: 'C', 3: 'G', 4: 'T', 0: '*'}
MATCH_PENALTY = 4
MISMATCH_PENALTY = 6
GAP_PENALTY = 8
GAP_EXTEND_PENALTY = 2
MIN_SEQUENCE_REQUIRED_FOR_MULTITHREADING = 1


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


def get_confident_positions(alignment):
    cigar_string = alignment.cigar_string.replace('=', 'M').replace('X', 'M')

    cigar_tuples = re.findall(r'(\d+)(\w)', cigar_string)

    grouped_tuples = list()
    prev_len = 0
    prev_op = None
    # group the matches together
    for cigar_len, cigar_op in cigar_tuples:
        if prev_op is None:
            prev_op = cigar_op
            prev_len = int(cigar_len)
        elif prev_op == cigar_op:
            # simply extend the operation
            prev_len += int(cigar_len)
        else:
            grouped_tuples.append((prev_op, prev_len))
            prev_op = cigar_op
            prev_len = int(cigar_len)
    if prev_op is not None:
        grouped_tuples.append((prev_op, prev_len))

    ref_index = alignment.reference_begin
    read_index = 0

    for cigar_op, cigar_len in grouped_tuples:
        if cigar_op == 'M' and cigar_len >= 5:
            return ref_index, read_index

        if cigar_op == 'S':
            read_index += cigar_len
        elif cigar_op == 'I':
            read_index += cigar_len
        elif cigar_op == 'D':
            ref_index += cigar_len
        elif cigar_op == 'M':
            ref_index += cigar_len
            read_index += cigar_len
        else:
            raise ValueError(TextColor.RED + "ERROR: INVALID CIGAR OPERATION ENCOUNTERED WHILTE STITCHING: "
                             + str(cigar_op) + "\n")

    return -1, -1


def alignment_stitch(sequence_chunks):
    sequence_chunks = sorted(sequence_chunks, key=lambda element: (element[1], element[2]))
    contig, running_start, running_end, running_sequence = sequence_chunks[0]
    # if len(running_sequence) < 500:
    #     sys.stderr.write("ERROR: CURRENT SEQUENCE LENGTH TOO SHORT: " + sequence_chunk_keys[0] + "\n")
    #     exit()

    aligner = PEPPER.Aligner(MATCH_PENALTY, MISMATCH_PENALTY, GAP_PENALTY, GAP_EXTEND_PENALTY)
    filter = PEPPER.Filter()
    for i in range(1, len(sequence_chunks)):
        _, this_start, this_end, this_sequence = sequence_chunks[i]
        if this_start < running_end:
            # overlap
            overlap_bases = running_end - this_start
            overlap_bases = overlap_bases + int(overlap_bases * BASE_ERROR_RATE)

            reference_sequence = running_sequence[-overlap_bases:]
            read_sequence = this_sequence[:overlap_bases]

            alignment = PEPPER.Alignment()
            aligner.SetReferenceSequence(reference_sequence, len(reference_sequence))
            aligner.Align_cpp(read_sequence, filter, alignment, 0)

            if alignment.best_score == 0:
                # we are going to say that the left sequence is right
                left_sequence = running_sequence

                # we are going to  say right sequence is also right
                right_sequence = this_sequence

                # but there are 10 'N's as overlaps
                overlap_sequence = 10 * 'N'
                # now append all three parts and we have a contiguous sequence
                running_sequence = left_sequence + overlap_sequence + right_sequence
                running_end = this_end
            else:
                pos_a, pos_b = get_confident_positions(alignment)

                if pos_a == -1 or pos_b == -1:
                    # we are going to say that the left sequence is right
                    left_sequence = running_sequence

                    # we are going to  say right sequence is also right
                    right_sequence = this_sequence

                    # but there are 10 'N's as overlaps
                    overlap_sequence = 10 * 'N'

                    # now append all three parts and we have a contiguous sequence
                    running_sequence = left_sequence + overlap_sequence + right_sequence
                    running_end = this_end
                else:
                    # this is a perfect match so we can simply stitch them
                    # take all of the sequence from the left
                    left_sequence = running_sequence[:-overlap_bases]
                    # get the bases that overlapped
                    overlap_sequence = reference_sequence[:pos_a]
                    # get sequences from current sequence
                    right_sequence = this_sequence[pos_b:]

                    # now append all three parts and we have a contiguous sequence
                    running_sequence = left_sequence + overlap_sequence + right_sequence
                    running_end = this_end
        else:
            # this means there was a gap before this chunk, which could be low read coverage in a small contig.
            running_sequence = running_sequence + this_sequence
            running_end = this_end

    return contig, running_start, running_end, running_sequence


def get_candidates(reference_sequence, read_sequence, start_pos, end_pos, hp_tag):
    aligner = PEPPER.Aligner(MATCH_PENALTY, MISMATCH_PENALTY, GAP_PENALTY, GAP_EXTEND_PENALTY)
    alignment_filter = PEPPER.Filter()
    alignment = PEPPER.Alignment()
    aligner.SetReferenceSequence(reference_sequence, len(reference_sequence))
    aligner.Align_cpp(read_sequence, alignment_filter, alignment, 0)

    cigar_string = alignment.cigar_string.replace('=', 'M')
    cigar_tuples = re.findall(r'(\d+)(\w)', cigar_string)
    read_index = 0
    ref_index = alignment.reference_begin
    ref_pos = start_pos + alignment.reference_begin

    candidate_list = []
    # group the matches together
    for tup_i, (cigar_len, cigar_op) in enumerate(cigar_tuples):
        cigar_len = int(cigar_len)
        # soft clip
        if cigar_op == 'S':
            read_index += cigar_len
        # match
        elif cigar_op == 'M':
            for i in range(0, cigar_len):
                # look forward if the next operation is an insert or delete
                if i == cigar_len - 1 and tup_i < len(cigar_tuples) - 1:
                    if cigar_tuples[tup_i+1][1] == 'I' or cigar_tuples[tup_i+1][1] == 'D':
                        read_index += 1
                        ref_index += 1
                        ref_pos += 1
                        continue

                ref_base = reference_sequence[ref_index]
                read_base = read_sequence[read_index]
                if ref_base != read_base:
                    # start_pos, end_pos, ref_allele, read_allele, type[0: snp, 1: IN, 2: del]
                    candidate_list.append((ref_pos, ref_pos + 1, ref_base, read_base, 0))
                read_index += 1
                ref_index += 1
                ref_pos += 1
        # mismatch
        elif cigar_op == 'X':
            for i in range(0, cigar_len):
                # look forward if the next operation is an insert or delete
                if i == cigar_len - 1 and tup_i < len(cigar_tuples) - 1:
                    if cigar_tuples[tup_i+1][1] == 'I' or cigar_tuples[tup_i+1][1] == 'D':
                        read_index += 1
                        ref_index += 1
                        ref_pos += 1
                        continue

                ref_base = reference_sequence[ref_index]
                read_base = read_sequence[read_index]
                if ref_base != read_base:
                    # start_pos, end_pos, ref_allele, read_allele, type[0: snp, 1: IN, 2: del]
                    candidate_list.append((ref_pos, ref_pos + 1, ref_base, read_base, hp_tag, 'SNP'))
                read_index += 1
                ref_index += 1
                ref_pos += 1
        # insert
        elif cigar_op == 'I':
            if ref_index > 0 and read_index > 0:
                ref_allele = reference_sequence[ref_index - 1]
                insert_allele = read_sequence[read_index - 1]
                for i in range(0, cigar_len):
                    insert_allele = insert_allele + read_sequence[read_index]
                    read_index += 1

                # start_pos, end_pos, ref_allele, read_allele, type[0: snp, 1: IN, 2: del]
                candidate_list.append((ref_pos - 1, ref_pos - 1 + 1, ref_allele, insert_allele, hp_tag, 'IN'))
            else:
                read_index += cigar_len

        # delete
        elif cigar_op == 'D':
            if ref_index > 0 and read_index > 0:
                ref_allele = reference_sequence[ref_index - 1]
                delete_allele = read_sequence[read_index - 1]
                start_pos = ref_pos - 1
                for i in range(0, cigar_len):
                    ref_allele = ref_allele + reference_sequence[ref_index]
                    ref_index += 1
                    ref_pos += 1
                candidate_list.append((start_pos, start_pos + cigar_len + 1, ref_allele, delete_allele, hp_tag, 'DEL'))
            else:
                ref_index += cigar_len
                ref_pos += cigar_len
        else:
            print("UNDEFINED CIGAR: ", cigar_op, cigar_len)
    return candidate_list


def group_adjacent_mismatches(mismatches):
    all_groups = []
    current_group = []
    for mismatch in mismatches:
        if len(current_group) == 0:
            current_group.append(mismatch)
        elif abs(current_group[-1][0] - mismatch[0]) <= 1:
            current_group.append(mismatch)
        else:
            all_groups.append(current_group)
            current_group = [mismatch]
    all_groups.append(current_group)

    return all_groups


def mismatch_groups_to_variants(mismatch_group):
    ref_dict = defaultdict()
    allele_dict = defaultdict()
    all_positions = set()

    mismatch_group = sorted(mismatch_group, key=operator.itemgetter(0, 1, 4))
    for pos, indx, ref_base, alt, hp_tag in mismatch_group:
        all_positions.add((pos, indx))
        ref_dict[(pos, indx)] = ref_base
        allele_dict[(pos, indx, hp_tag)] = alt

    all_positions = list(all_positions)
    all_positions = sorted(all_positions, key=operator.itemgetter(0, 1))
    start_pos = all_positions[0][0]
    start_indx = all_positions[0][1]
    end_pos = all_positions[-1][0]
    start_ref = ref_dict[(start_pos, start_indx)]

    if start_indx > 0 or start_ref == 0:
        # print("GROUP ERROR: ", mismatch_group)
        return None

    ref_allele = []
    alt_allele_1 = []
    alt_allele_2 = []
    for pos, indx in all_positions:
        ref_base = ref_dict[(pos, indx)]
        ref_allele.append(ref_base)
        if (pos, indx, 1) in allele_dict.keys():
            alt_allele_1.append(allele_dict[(pos, indx, 1)])
        else:
            alt_allele_1.append(ref_base)

        if (pos, indx, 2) in allele_dict.keys():
            alt_allele_2.append(allele_dict[(pos, indx, 2)])
        else:
            alt_allele_2.append(ref_base)

    ref_seq = ''.join(label_decoder[i] for i in ref_allele)
    alt1_seq = ''.join(label_decoder[i] for i in alt_allele_1)
    alt2_seq = ''.join(label_decoder[i] for i in alt_allele_2)

    if alt1_seq == ref_seq:
        alt1_seq = ''
    if alt2_seq == ref_seq:
        alt2_seq = ''

    v_type = 'SNP'
    if len(ref_seq) > 1:
        v_type = 'INDEL'

    return start_pos, end_pos+1, ref_seq, alt1_seq, alt2_seq, v_type


def get_anchor_positions(base_predictions, ref_seq, indices, positions):
    is_diff = base_predictions != ref_seq
    major_positions = np.where(indices == 0)
    minor_positions = np.where(indices != 0)
    delete_anchors = positions[major_positions][np.where(base_predictions[major_positions] == 0)] - 1
    insert_anchors = positions[minor_positions][np.where(base_predictions[minor_positions] != 0)]

    return list(delete_anchors), list(insert_anchors)


def small_chunk_stitch(file_name, reference_file_path, contig, small_chunk_keys):
    # for chunk_key in small_chunk_keys:
    fasta_handler = PEPPER.FASTA_handler(reference_file_path)
    start_time = time.time()

    all_mismatches_h1 = list()
    all_mismatches_h2 = list()
    all_h1_positions = list()
    all_h2_positions = list()
    all_h1_anchors = list()
    all_h2_anchors = list()
    position_to_ref_base = defaultdict()
    # Find candidates per chunk
    for contig_name, _st, _end in small_chunk_keys:
        all_positions = set()
        highest_index_per_pos = defaultdict(lambda: 0)

        chunk_name = contig_name + '-' + str(_st) + '-' + str(_end)

        with h5py.File(file_name, 'r') as hdf5_file:
            smaller_chunks = set(hdf5_file['predictions'][contig][chunk_name].keys()) - {'contig_start', 'contig_end'}

        smaller_chunks = sorted(smaller_chunks)

        for chunk in smaller_chunks:
            with h5py.File(file_name, 'r') as hdf5_file:
                bases_h1 = hdf5_file['predictions'][contig][chunk_name][chunk]['bases_h1'][()]
                bases_h2 = hdf5_file['predictions'][contig][chunk_name][chunk]['bases_h2'][()]
                positions = hdf5_file['predictions'][contig][chunk_name][chunk]['position'][()]
                indices = hdf5_file['predictions'][contig][chunk_name][chunk]['index'][()]
                ref_seq = hdf5_file['predictions'][contig][chunk_name][chunk]['ref_seq'][()]
                coverage = hdf5_file['predictions'][contig][chunk_name][chunk]['coverage'][()]

            positions = np.array(positions, dtype=np.int64)
            base_predictions_h1 = np.array(bases_h1, dtype=np.int)
            base_predictions_h2 = np.array(bases_h2, dtype=np.int)
            ref_seq = np.array(ref_seq, dtype=np.int)
            delete_anchors_h1, insert_anchors_h1 = get_anchor_positions(base_predictions_h1, ref_seq, indices, positions)
            delete_anchors_h2, insert_anchors_h2 = get_anchor_positions(base_predictions_h2, ref_seq, indices, positions)

            all_anchors_h1 = insert_anchors_h1 + delete_anchors_h1
            all_anchors_h2 = insert_anchors_h2 + delete_anchors_h2
            all_h1_anchors.extend(all_anchors_h1)
            all_h2_anchors.extend(all_anchors_h2)

            # as I have carried the reference sequence over, we will get the candidates naturally
            for pos, indx, ref_base, base_pred_h1, base_pred_h2, cov_h1, cov_h2 in zip(positions,
                                                                       indices,
                                                                       ref_seq,
                                                                       base_predictions_h1,
                                                                       base_predictions_h2,
                                                                       coverage[1],
                                                                       coverage[2]):
                if indx < 0 or pos < 0:
                    continue
                if indx == 0:
                    position_to_ref_base[pos] = ref_base

                if (pos, indx) not in all_positions:
                    if ref_base != base_pred_h1:
                        if cov_h1 >= 2:
                            all_mismatches_h1.append((pos, indx, ref_base, base_pred_h1))
                            if indx == 0:
                                all_h1_positions.append(pos)
                    if ref_base != base_pred_h2:
                        if cov_h2 >= 2:
                            all_mismatches_h2.append((pos, indx, ref_base, base_pred_h2))
                            if indx == 0:
                                all_h2_positions.append(pos)
                    all_positions.add((pos, indx))
                    highest_index_per_pos[pos] = max(highest_index_per_pos[pos], indx)

    # now add all the anchor positions
    for pos in all_h1_anchors:
        if pos not in position_to_ref_base.keys():
            continue
        ref_base = position_to_ref_base[pos]
        if pos not in all_h1_positions:
            all_mismatches_h1.append((pos, 0, ref_base, ref_base))
            all_h1_positions.append(pos)

    for pos in all_h2_anchors:
        if pos not in position_to_ref_base.keys():
            continue
        ref_base = position_to_ref_base[pos]
        if pos not in all_h2_positions:
            all_mismatches_h2.append((pos, 0, ref_base, ref_base))
            all_h2_positions.append(pos)

    # sys.stderr.write("TIME ELAPSED: " + str(time.time() - start_time) + "\n")
    return contig, all_mismatches_h1, all_mismatches_h2


def find_candidates(hdf5_file_path,  reference_file_path, contig, sequence_chunk_keys, threads):
    sequence_chunk_keys = sorted(sequence_chunk_keys)
    sequence_chunk_key_list = list()
    for sequence_chunk_key in sequence_chunk_keys:
        contig, st, end = sequence_chunk_key.split('-')
        sequence_chunk_key_list.append((contig, int(st), int(end)))

    sequence_chunk_key_list = sorted(sequence_chunk_key_list, key=lambda element: (element[1], element[2]))

    all_mismatches = list()
    all_positions_h1 = set()
    all_positions_h2 = set()
    # generate the dictionary in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        file_chunks = chunks(sequence_chunk_key_list, max(MIN_SEQUENCE_REQUIRED_FOR_MULTITHREADING,
                                                          int(len(sequence_chunk_key_list) / threads) + 1))

        futures = [executor.submit(small_chunk_stitch, hdf5_file_path, reference_file_path, contig, file_chunk)
                   for file_chunk in file_chunks]
        for fut in concurrent.futures.as_completed(futures):
            if fut.exception() is None:
                contig, mismatches_h1, mismatches_h2 = fut.result()

                sys.stderr.write("UPDATING DICTIONARY\n")
                for pos, indx, ref, alt in mismatches_h1:
                    if (pos, indx) not in all_positions_h1:
                        all_positions_h1.add((pos, indx))
                        all_mismatches.append((pos, indx, ref, alt, 1))

                for pos, indx, ref, alt in mismatches_h2:
                    if (pos, indx) not in all_positions_h2:
                        all_positions_h2.add((pos, indx))
                        all_mismatches.append((pos, indx, ref, alt, 2))
            else:
                sys.stderr.write("ERROR IN THREAD: " + str(fut.exception()) + "\n")
            fut._result = None  # python issue 27144

    all_groups = group_adjacent_mismatches(sorted(all_mismatches, key=operator.itemgetter(0, 1, 4)))

    all_candidate_positions = set()
    candidate_positional_map_h1 = defaultdict()
    candidate_positional_map_h2 = defaultdict()

    for mismatch_group in all_groups:
        if not mismatch_group:
            continue
        # print("H1", mismatch_group)
        variant = mismatch_groups_to_variants(mismatch_group)

        if variant is not None:
            _st, _end, ref, alt1, alt2, v_type = variant
        else:
            continue
        # print(_st, _end, ref, alt1, alt2, v_type)
        if len(alt1) >= 1:
            all_candidate_positions.add(_st)
            candidate_positional_map_h1[_st] = (_st, _end, ref, alt1, v_type)
        if len(alt2) >= 1:
            all_candidate_positions.add(_st)
            candidate_positional_map_h2[_st] = (_st, _end, ref, alt2, v_type)

    return all_candidate_positions, candidate_positional_map_h1, candidate_positional_map_h2
