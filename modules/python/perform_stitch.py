import h5py
import sys
import re
from modules.python.TextColor import TextColor
from modules.python.StitchV2 import create_consensus_sequence
from os.path import isfile, join
from pathlib import Path
from os import listdir


def get_file_paths_from_directory(directory_path):
    """
    Returns all paths of files given a directory path
    :param directory_path: Path to the directory
    :return: A list of paths of files
    """
    file_paths = [join(directory_path, file) for file in listdir(directory_path) if isfile(join(directory_path, file))
                  and file[-3:] == 'hdf']
    return file_paths


def number_key(name):
    """
    Sorting: https://stackoverflow.com/questions/4287209/sort-list-of-strings-by-integer-suffix-in-python
    :param name:
    :return:
    """
    parts = re.findall('[^0-9]+|[0-9]+', name)
    L = []
    for part in parts:
        try:
            L.append(int(part))
        except ValueError:
            L.append(part)
    return L


def perform_stitch(hdf_file_path, output_path, threads):
    all_prediction_files = get_file_paths_from_directory(hdf_file_path)

    all_contigs = set()
    # get contigs from all of the files
    for prediction_file in sorted(all_prediction_files):
        with h5py.File(prediction_file, 'r') as hdf5_file:
            contigs = list(hdf5_file['predictions'].keys())
            all_contigs.update(contigs)

    output_directory = Path(output_path).resolve().parents[0]
    output_directory.mkdir(parents=True, exist_ok=True)

    consensus_fasta_file = open(output_path + '_pepper.fa', 'w')

    for contig in sorted(all_contigs):
        sys.stderr.write(TextColor.YELLOW + "INFO: PROCESSING CONTIG: " + contig + "\n" + TextColor.END)

        # get all the chunk keys from all the files
        all_chunk_keys = list()
        for prediction_file in all_prediction_files:
            with h5py.File(prediction_file, 'r') as hdf5_file:
                if contig not in hdf5_file['predictions'].keys():
                    continue
                chunk_keys = sorted(hdf5_file['predictions'][contig].keys())
                for chunk_key in chunk_keys:
                    all_chunk_keys.append((prediction_file, chunk_key))

        consensus_sequence = create_consensus_sequence(contig,
                                                       all_chunk_keys,
                                                       threads)
        sys.stderr.write(TextColor.BLUE + "INFO: FINISHED PROCESSING " + contig + ", POLISHED SEQUENCE LENGTH: "
                         + str(len(consensus_sequence)) + ".\n" + TextColor.END)

        # TODO: I should write a FASTA handler here. This is too sloppy.
        if consensus_sequence is not None and len(consensus_sequence) > 0:
            consensus_fasta_file.write('>' + contig + "\n")
            consensus_fasta_file.write(consensus_sequence+"\n")

    hdf5_file.close()
