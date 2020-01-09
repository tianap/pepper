from pysam import VariantFile, VariantHeader
from build import PEPPER
import time
import collections
Candidate = collections.namedtuple('Candidate', 'chromosome_name pos_start pos_end ref '
                                                'alternate_alleles allele_depths '
                                                'allele_frequencies genotype qual gq predictions')

class VCFWriter:
    def __init__(self, reference_file_path, contigs, sample_name, output_dir, filename):
        self.fasta_handler = PEPPER.FASTA_handler(reference_file_path)
        self.contigs = contigs
        vcf_header = self.get_vcf_header(sample_name)
        time_str = time.strftime("%m%d%Y_%H%M%S")

        self.vcf_file = VariantFile(output_dir + filename + '.vcf', 'w', header=vcf_header)

    def write_vcf_records(self, called_variant):
        contig, ref_start, ref_end, ref_seq, alleles, genotype = called_variant
        alleles = tuple([ref_seq]) + tuple(alleles)
        # print(str(chrm), st_pos, end_pos, qual, rec_filter, alleles, genotype, gq)
        vcf_record = self.vcf_file.new_record(contig=str(contig), start=ref_start,
                                              stop=ref_end, id='.', qual=60,
                                              filter='PASS', alleles=alleles, GT=genotype, GQ=60)
        self.vcf_file.write(vcf_record)

    def get_vcf_header(self, sample_name):
        header = VariantHeader()
        items = [('ID', "PASS"),
                 ('Description', "All filters passed")]
        header.add_meta(key='FILTER', items=items)
        items = [('ID', "refCall"),
                 ('Description', "Call is homozygous")]
        header.add_meta(key='FILTER', items=items)
        items = [('ID', "lowGQ"),
                 ('Description', "Low genotype quality")]
        header.add_meta(key='FILTER', items=items)
        items = [('ID', "lowQUAL"),
                 ('Description', "Low variant call quality")]
        header.add_meta(key='FILTER', items=items)
        items = [('ID', "conflictPos"),
                 ('Description', "Overlapping record")]
        header.add_meta(key='FILTER', items=items)
        items = [('ID', "GT"),
                 ('Number', 1),
                 ('Type', 'String'),
                 ('Description', "Genotype")]
        header.add_meta(key='FORMAT', items=items)
        items = [('ID', "GQ"),
                 ('Number', 1),
                 ('Type', 'Float'),
                 ('Description', "Genotype Quality")]
        header.add_meta(key='FORMAT', items=items)

        for sq in self.contigs:
            contig_name = sq
            contig_length = self.fasta_handler.get_chromosome_sequence_length(sq)
            items = [('ID', contig_name),
                     ('length', contig_length)]
            header.add_meta(key='contig', items=items)

        header.add_sample(sample_name)

        return header
