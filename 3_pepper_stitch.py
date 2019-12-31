import h5py
import argparse
import sys
from modules.python.TextColor import TextColor
from modules.python.StitchV2 import create_consensus_sequence
from modules.python.VcfWriter import VCFWriter
from build import PEPPER


def perform_stitch(hdf_file_path, reference_path, output_path, threads, sample_name):
    with h5py.File(hdf_file_path, 'r') as hdf5_file:
        contigs = list(hdf5_file['predictions'].keys())

    vcf_file = VCFWriter(reference_path, sample_name, output_path)

    for contig in contigs:
        sys.stderr.write(TextColor.YELLOW + "INFO: PROCESSING CONTIG: " + contig + "\n" + TextColor.END)

        with h5py.File(hdf_file_path, 'r') as hdf5_file:
            chunk_keys = sorted(hdf5_file['predictions'][contig].keys())

        all_candidates = create_consensus_sequence(hdf_file_path, reference_path, contig, chunk_keys, threads)
        sys.stderr.write(TextColor.BLUE + "INFO: FINISHED PROCESSING " + contig + ", TOTAL CANDIDATES FOUND: "
                         + str(len(all_candidates)) + ".\n" + TextColor.END)
        vcf_file.write_vcf_records(contig, all_candidates)


    hdf5_file.close()


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
    '''
    parser = argparse.ArgumentParser(description="3_pepper_stitch.py performs the final stitching to generate  "
                                                 "the polished sequences.")
    parser.add_argument(
        "-i",
        "--input_hdf",
        type=str,
        required=True,
        help="Input hdf prediction file."
    )
    parser.add_argument(
        "-r",
        "--input_reference",
        type=str,
        required=True,
        help="Input reference/assembly file."
    )
    parser.add_argument(
        "-s",
        "--sample_name",
        type=str,
        required=True,
        help="Name of the sample."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory."
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=5,
        help="Number of threads."
    )

    FLAGS, unparsed = parser.parse_known_args()
    perform_stitch(FLAGS.input_hdf, FLAGS.input_reference, FLAGS.output_dir, FLAGS.threads, FLAGS.sample_name)
