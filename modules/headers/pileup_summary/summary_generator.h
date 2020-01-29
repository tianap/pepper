//
// Created by Kishwar Shafin on 25/2/19.
//

#ifndef HELEN_SUMMARY_GENERATOR_H
#define HELEN_SUMMARY_GENERATOR_H

#include <math.h>
#include <algorithm>
#include <iomanip>
#include <assert.h>
using namespace std;
#include "../dataio/bam_handler.h"


namespace ImageOptions {
    static constexpr int MAX_COLOR_VALUE = 254;
};

class Position {
    public:
        long long position_ref;
        int position_index;
        int feature_index;
        int hp_tag;

        Position(long long position_ref, int position_index, int feature_index) {
            this->position_ref = position_ref;
            this->position_index = position_index;
            this->feature_index = feature_index;
            this->hp_tag = 0;
        }

        Position(long long position_ref, int position_index, int feature_index, int hp_tag) {
            this->position_ref = position_ref;
            this->position_index = position_index;
            this->feature_index = feature_index;
            this->hp_tag = hp_tag;
        }


        bool operator<(const Position& that)const
        {
            if(this->position_ref == that.position_ref) {
                if(this->position_index == that.position_index) {
                    if(this->feature_index == that.feature_index) {
                        return this->hp_tag < that.hp_tag;
                    }
                    else{
                        return this->feature_index < that.feature_index;
                    }
                }
                else {
                    return this->position_index < that.position_index;
                }
            } else {
                return this->position_ref < that.position_ref;
            }
        }

};

class SummaryGenerator {
    long long ref_start;
    long long ref_end;
    string chromosome_name;
    string reference_sequence;

    map<Position, double> summaries;
    map<long long, int> longest_insert_count;
    map<pair<int, long long>, double> coverage;

    map<Position, char> label_map;
public:
    vector< vector< vector<uint8_t> > > image;
    vector<uint8_t> ref_image;
    vector< vector<uint8_t> > labels;
    vector<pair<long long, int> > genomic_pos;
    vector<int> bad_label_positions;
    vector< vector<int> > coverage_count;

    SummaryGenerator(string reference_sequence,
                     string chromosome_name,
                     long long ref_start,
                     long long ref_end);
    void generate_summary(vector <type_read> &reads,
                          long long start_pos,
                          long long end_pos);

    void generate_train_summary(vector <type_read> &reads,
                                long long start_pos,
                                long long end_pos,
                                type_read truth_read_h1,
                                type_read truth_read_h2);

    void iterate_over_read(type_read read, long long region_start, long long region_end);
    int get_sequence_length(long long start_pos, long long end_pos);
    void generate_labels(type_read read, int hp_tag, long long region_start, long long region_end);
    void generate_ref_features();
    void debug_print(long long start_pos, long long end_pos, bool print_label);
    void generate_image(long long start_pos, long long end_pos);
};


#endif //HELEN_SUMMARY_GENERATOR_H
