//
// Created by Kishwar Shafin on 11/1/18.
//

#include "../../headers/pileup_summary/summary_generator.h"

SummaryGenerator::SummaryGenerator(string reference_sequence, string chromosome_name, long long ref_start,
                                   long long ref_end) {
    this->reference_sequence = reference_sequence;
    this->ref_start = ref_start;
    this->ref_end = ref_end;
    this->chromosome_name = chromosome_name;
}


int get_feature_index(char base, bool is_reverse) {
    base = toupper(base);
    if (is_reverse) {
        if (base == 'A') return 0;
        if (base == 'C') return 1;
        if (base == 'G') return 2;
        if (base == 'T') return 3;
        return 8;
    } else {
        // tagged and forward
        if (base == 'A') return 4;
        if (base == 'C') return 5;
        if (base == 'G') return 6;
        if (base == 'T') return 7;
        return 9;
    }
}


int get_reference_feature_index(char base) {
    base = toupper(base);
    if (base == 'A') return 1;
    if (base == 'C') return 2;
    if (base == 'G') return 3;
    if (base == 'T') return 4;
    return 0;
}


uint8_t get_labels(char base) {
    base = toupper(base);
    if (base == 'A') return 1;
    if (base == 'C') return 2;
    if (base == 'G') return 3;
    if (base == 'T') return 4;
    if (base == '*') return 0; // this is for deleted bases, but the number is so small that it creates confusion
    if (base == '#') return 0;
    return 0;
}


void SummaryGenerator::iterate_over_read(type_read read, long long region_start, long long region_end) {
    int read_index = 0;
    long long ref_position = read.pos;
    int cigar_index = 0;
    int base_quality = 0;
    long long reference_index;
    int hp_tag = read.hp_tag;

    for (auto &cigar: read.cigar_tuples) {
        if (ref_position > region_end) break;
        switch (cigar.operation) {
            case CIGAR_OPERATIONS::EQUAL:
            case CIGAR_OPERATIONS::DIFF:
            case CIGAR_OPERATIONS::MATCH:
                cigar_index = 0;
                if (ref_position < ref_start) {
                    cigar_index = min(ref_start - ref_position, (long long) cigar.length);
                    read_index += cigar_index;
                    ref_position += cigar_index;
                }
                for (int i = cigar_index; i < cigar.length; i++) {
                    reference_index = ref_position - ref_start;
                    //read.base_qualities[read_index] base quality
                    if (ref_position >= ref_start && ref_position <= ref_end) {
                        char base = read.sequence[read_index];

                        // update the summary of base, hp_tag not present
                        Position p(ref_position, 0, get_feature_index(base, read.flags.is_reverse));
                        summaries[p] += 1.0;
                        coverage[make_pair(ref_position, 0)] += 1.0;

                        // update the summary of base if hp_tag is present
                        if(hp_tag != 0) {
                            Position ph(ref_position, 0, get_feature_index(base, read.flags.is_reverse), hp_tag);
                            summaries[ph] += 1.0;
                            coverage[make_pair(ref_position, hp_tag)] += 1.0;
                        }
                    }
                    read_index += 1;
                    ref_position += 1;
                }
                break;
            case CIGAR_OPERATIONS::IN:
//                base_qualities = read.base_qualities.begin() + read_index, read.base_qualities.begin() + (read_index + cigar.length);
                reference_index = ref_position - ref_start - 1;

                if (ref_position - 1 >= ref_start &&
                    ref_position - 1 <= ref_end) {
                    // process insert allele here
                    string alt;
                    alt = read.sequence.substr(read_index, cigar.length);
                    for (int i = 0; i < cigar.length; i++) {
                        pair<long long, int> position_pair = make_pair(ref_position - 1, i);
                        Position p(ref_position - 1, i+1, get_feature_index(alt[i], read.flags.is_reverse));
                        summaries[p] += 1.0;

                        // update the summary of base if hp_tag is present
                        if(hp_tag != 0) {
                            Position ph(ref_position - 1, i+1, get_feature_index(alt[i], read.flags.is_reverse), hp_tag);
                            summaries[ph] += 1.0;
                        }
                    }
                    longest_insert_count[ref_position - 1] = std::max(longest_insert_count[ref_position - 1],
                                                                      (int) alt.length());
                }
                read_index += cigar.length;
                break;
            case CIGAR_OPERATIONS::REF_SKIP:
            case CIGAR_OPERATIONS::PAD:
            case CIGAR_OPERATIONS::DEL:
//                base_quality = read.base_qualities[max(0, read_index)];
                reference_index = ref_position - ref_start - 1;
                // process delete allele here
                for (int i = 0; i < cigar.length; i++) {
                    if (ref_position + i >= ref_start && ref_position + i <= ref_end) {
                        // update the summary of base
                        Position p(ref_position + i, 0, get_feature_index('*', read.flags.is_reverse));
                        summaries[p] += 1.0;
                        coverage[make_pair(ref_position, 0)] += 1.0;

                        // update the summary of base if hp_tag is present
                        if(hp_tag != 0) {
                            Position ph(ref_position + i, 0, get_feature_index('*', read.flags.is_reverse), hp_tag);
                            summaries[ph] += 1.0;
                            coverage[make_pair(ref_position + i, hp_tag)] += 1.0;
                        }
                    }
                }
                ref_position += cigar.length;
                break;
            case CIGAR_OPERATIONS::SOFT_CLIP:
                read_index += cigar.length;
                break;
            case CIGAR_OPERATIONS::HARD_CLIP:
                break;
        }
    }
}

//int SummaryGenerator::get_sequence_length(long long start_pos, long long end_pos) {
//    int length = 0;
//    for (int i = start_pos; i <= end_pos; i++) {
//        length += 1;
//        if (longest_insert_count[i] > 0) {
//            length += longest_insert_count[i];
//        }
//    }
//    return length;
//}

bool check_base(char base) {
    if(base=='A' || base=='a' ||
       base=='C' || base=='c' ||
       base=='T' || base=='t' ||
       base =='G' || base=='g' || base == '*' || base == '#') return true;
    return false;
}

void SummaryGenerator::generate_labels(type_read read, int hp_tag, long long region_start, long long region_end) {
    int read_index = 0;
    long long ref_position = read.pos;
    int cigar_index = 0;
    int base_quality = 0;
    long long reference_index;

    for (auto &cigar: read.cigar_tuples) {
        if (ref_position > region_end) break;
        switch (cigar.operation) {
            case CIGAR_OPERATIONS::EQUAL:
            case CIGAR_OPERATIONS::DIFF:
            case CIGAR_OPERATIONS::MATCH:
                cigar_index = 0;
                if (ref_position < ref_start) {
                    cigar_index = min(ref_start - ref_position, (long long) cigar.length);
                    read_index += cigar_index;
                    ref_position += cigar_index;
                }
                for (int i = cigar_index; i < cigar.length; i++) {
                    reference_index = ref_position - ref_start;
//                    cout<<ref_position<<" "<<ref_end<<" "<<region_end<<endl;
                    if (ref_position >= ref_start && ref_position <= ref_end) {
                        char base = read.sequence[read_index];
                        Position p(ref_position, 0, 0, hp_tag);
                        label_map[p] = base;
                    }
                    read_index += 1;
                    ref_position += 1;
                }
                break;
            case CIGAR_OPERATIONS::IN:
                reference_index = ref_position - ref_start - 1;
                if (ref_position - 1 >= ref_start &&
                    ref_position - 1 <= ref_end) {
                    // process insert allele here
                    string alt;
                    alt = read.sequence.substr(read_index, cigar.length);

                    for (int i = 0; i < longest_insert_count[ref_position - 1]; i++) {
                        char base = '#';
                        if (i < alt.length()) {
                            base = alt[i];
                        }
                        Position ph(ref_position - 1, i+1, 0, hp_tag);
                        label_map[ph] = base;
                    }
                }
                read_index += cigar.length;
                break;
            case CIGAR_OPERATIONS::REF_SKIP:
            case CIGAR_OPERATIONS::PAD:
            case CIGAR_OPERATIONS::DEL:
                reference_index = ref_position - ref_start - 1;

                if (ref_position >= ref_start && ref_position <= ref_end) {
                    // process delete allele here
                    for (int i = 0; i < cigar.length; i++) {
                        if (ref_position + i >= ref_start && ref_position + i <= ref_end) {
                            // DELETE
                            char base = '*';
                            Position ph(ref_position + i, 0, 0, hp_tag);
                            label_map[ph] = '*';
                        }
                    }
                }
                ref_position += cigar.length;
                break;
            case CIGAR_OPERATIONS::SOFT_CLIP:
                read_index += cigar.length;
                break;
            case CIGAR_OPERATIONS::HARD_CLIP:
                break;
        }
    }
}


void SummaryGenerator::debug_print(long long start_pos, long long end_pos, bool print_label) {
    cout<<"HERE"<<endl;
    for (int hp_tag = 0; hp_tag < 3; hp_tag++) {
        cout<<"HP TAG: "<< hp_tag <<endl;
        cout << setprecision(3);
        for (int i = start_pos; i <= end_pos; i++) {
            if (i == start_pos) cout << "REF:\t";
            cout << "  " << reference_sequence[i - start_pos] << "\t";
            if (longest_insert_count[i] > 0) {
                for (int ii = 0; ii < longest_insert_count[i]; ii++)cout << "  *" << "\t";
            }
        }
        cout<<ref_image.size()<<endl;
        for (int i = 0; i < ref_image.size(); i++) {
            if (i == 0) cout << "REF2:\t";
            printf("%3d\t", ref_image[i]);
        }
        cout << endl;
        if(hp_tag > 0 && print_label) {
            for (int i = start_pos; i <= end_pos; i++) {
                if (i == start_pos) cout << "TRH:\t";
                Position p(i, 0, 0, hp_tag);
                cout << "  " << label_map[p] << "\t";
                if (longest_insert_count[i] > 0) {
                    for (int ii = 0; ii < longest_insert_count[i]; ii++) {
                        Position ph(i, ii + 1, 0, hp_tag);
                        if (label_map.find(ph) != label_map.end())
                            cout << "  " << label_map[ph] << "\t";
                        else cout << "  *" << "\t";
                    }
                }
            }
            cout << endl;

            for (int i = 0; i < labels[hp_tag-1].size(); i++) {
                if (i == 0) cout << "LBL:\t";
                printf("%3d\t", labels[hp_tag-1][i]);
            }
            cout << endl;
        }

        cout << "POS:\t";
        for (int i = 0; i < genomic_pos.size(); i++) {

//        cout << "(" << genomic_pos[i].first << "," << genomic_pos[i].second << ")\t";
            printf("%3lld\t", (genomic_pos[i].first) % 100);
        }
        cout << endl;

        cout << "-------------" << endl;
//        print(image.size())
//        print(image[0].size())
//        print(image[0][0].size())
        for (int i = 0; i < image[hp_tag][0].size(); i++) {
            if (i == 0)cout << "AFW:\t";
            if (i == 1)cout << "CFW:\t";
            if (i == 2)cout << "GFW:\t";
            if (i == 3)cout << "TFW:\t";
            if (i == 4)cout << "ARV:\t";
            if (i == 5)cout << "CRV:\t";
            if (i == 6)cout << "GRV:\t";
            if (i == 7)cout << "TRV:\t";
            if (i == 8)cout << "*FW:\t";
            if (i == 9)cout << "*RV:\t";

            for (int j = 0; j < image[hp_tag].size(); j++) {
                printf("%3d\t", image[hp_tag][j][i]);
            }
            cout << endl;
        }
        cout<<"---------------------------------------------------------"<<endl;
    }

}

void SummaryGenerator::generate_image(long long start_pos, long long end_pos) {
    // at this point labels and positions are generated, now generate the pileup
    for (int hp_tag = 0; hp_tag < 3; hp_tag++) {
        vector< vector<uint8_t> > image_hp;
        for (long long ref_pos = start_pos; ref_pos <= end_pos; ref_pos++) {
            vector <uint8_t> row;
            uint8_t pixel_value = 0;
            // iterate through the summaries
            for (int feature_index = 0; feature_index <= 9; feature_index++) {
                Position p(ref_pos, 0, feature_index, hp_tag);
                pixel_value = 0;
                if(summaries.find(p) != summaries.end())
                    pixel_value = (summaries[p] / max(1.0, coverage[make_pair(ref_pos, hp_tag)])) * ImageOptions::MAX_COLOR_VALUE;
                row.push_back(pixel_value);
            }
            assert(row.size() == 10);
            image_hp.push_back(row);

            if (longest_insert_count[ref_pos] > 0) {

                for (int ii = 0; ii < longest_insert_count[ref_pos]; ii++) {
                    vector <uint8_t> ins_row;

                    // iterate through the summaries
                    for (int feature_index = 0; feature_index <= 9; feature_index++) {
                        Position p(ref_pos, ii + 1, feature_index, hp_tag);
                        pixel_value = 0;
                        if(summaries.find(p) != summaries.end())
                            pixel_value = (summaries[p] / max(1.0, coverage[make_pair(ref_pos, hp_tag)])) * ImageOptions::MAX_COLOR_VALUE;
                        ins_row.push_back(pixel_value);

                    }
                    assert(ins_row.size() == 10);
                    image_hp.push_back(ins_row);
                }
            }
        }
        assert(image_hp.size() == genomic_pos.size());
        image.push_back(image_hp);
    }

}



void SummaryGenerator::generate_train_summary(vector <type_read> &reads,
                                              long long start_pos,
                                              long long end_pos,
                                              type_read truth_read_h1,
                                              type_read truth_read_h2) {
    for (auto &read:reads) {
        // this populates summaries
        if(read.mapping_quality >= 10) {
            iterate_over_read(read, start_pos, end_pos);
        }
    }

    // this populates base_labels and insert_labels dictionaries
    generate_labels(truth_read_h1, 1, start_pos, end_pos + 1);
    generate_labels(truth_read_h2, 2, start_pos, end_pos + 1);

    // after all the dictionaries are populated, we can simply walk through the region and generate a sequence
    for (long long pos = start_pos; pos <= end_pos; pos++) {
        genomic_pos.push_back(make_pair(pos, 0));
        if (longest_insert_count[pos] > 0) {
            for (int ii = 0; ii < longest_insert_count[pos]; ii++) {
                genomic_pos.push_back(make_pair(pos, ii + 1));
            }
        }
    }

    // after all the dictionaries are populated, we can simply walk through the region and generate a sequence
    for (int hp_tag = 1; hp_tag <=2 ; hp_tag++) {
        vector<uint8_t> hp_labels;
        for (long long pos = start_pos; pos <= end_pos; pos++) {
            Position p(pos, 0, 0, hp_tag);
            if (coverage[make_pair(pos, hp_tag)] > 0) {
                hp_labels.push_back(get_labels(label_map[p]));
            } else {
                hp_labels.push_back(get_labels('*'));
            }

            // if the label contains anything but ACTG
            if (!check_base(label_map[p])) {
//            cerr<<"INFO: INVALID REFERENCE BASE INDEX FOUND: ["<<chromosome_name<<":"<<start_pos<<"-"<<end_pos<<"] " <<
//                pos<<" "<<" "<<base_labels[pos]<<endl;
                bad_label_positions.push_back(hp_labels.size());
            }

            if (longest_insert_count[pos] > 0) {
                for (int ii = 0; ii < longest_insert_count[pos]; ii++) {
                    Position ph(pos, ii+1, 0, hp_tag);
                    if (label_map.find(ph) != label_map.end()) {
                        hp_labels.push_back(get_labels(label_map[ph]));

                        // if the label contains anything but ACTG
                        if (!check_base(label_map[ph])) {
//                        cerr<<"INFO: INVALID REFERENCE INSERT BASE INDEX FOUND: "<<chromosome_name<<" "<<
//                            pos<<" "<<insert_labels[make_pair(pos, ii)]<<endl;
                            bad_label_positions.push_back(labels.size());
                        }
                    } else hp_labels.push_back(get_labels('#'));
                }
            }
        }
        bad_label_positions.push_back(hp_labels.size());
        assert(hp_labels.size() == genomic_pos.size());

        labels.push_back(hp_labels);
    }

    // generate reference sequence
    for (int i = start_pos; i <= end_pos; i++) {
        ref_image.push_back(get_reference_feature_index(reference_sequence[i - start_pos]));

        if (longest_insert_count[i] > 0) {
            for (int ii = 0; ii < longest_insert_count[i]; ii++)
                ref_image.push_back(get_reference_feature_index('*'));
        }
    }
    generate_image(start_pos, end_pos);
//     at this point everything should be generated
//    debug_print(start_pos, end_pos, 1);
}


void SummaryGenerator::generate_summary(vector <type_read> &reads,
                                        long long start_pos,
                                        long long end_pos) {
    for (auto &read:reads) {
        // this populates summaries
        if(read.mapping_quality >= 10) {
            iterate_over_read(read, start_pos, end_pos);
        }
    }

    // after all the dictionaries are populated, we can simply walk through the region and generate a sequence
    for (long long pos = start_pos; pos <= end_pos; pos++) {
        genomic_pos.push_back(make_pair(pos, 0));
        if (longest_insert_count[pos] > 0) {
            for (int ii = 0; ii < longest_insert_count[pos]; ii++) {
                genomic_pos.push_back(make_pair(pos, ii + 1));
            }
        }
    }

    // generate reference sequence
    for (int i = start_pos; i <= end_pos; i++) {
        ref_image.push_back(get_reference_feature_index(reference_sequence[i - start_pos]));

        if (longest_insert_count[i] > 0) {
            for (int ii = 0; ii < longest_insert_count[i]; ii++)
                ref_image.push_back(get_reference_feature_index('*'));
        }
    }

    generate_image(start_pos, end_pos);

    //at this point everything should be generated
//    debug_print(start_pos, end_pos, 0);
}