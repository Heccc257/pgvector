#pragma once
#include <stdlib.h>
#include <stdint.h>

typedef struct{
    int d;
    int m;
    int nbits;
    int code_nums;
    int d_pq;
    int tuple_id;
    size_t table_size;
    uint8_t *codes;
    float* centroids;
    float* pq_dist_cache_data;
    float* qdata;
    bool use_cache;

}PQDist;

void pqdist_init(PQDist* pqdist, int _d, int _m, int _nbits);
void PQDist_load(PQDist* pq, const char* filename);
void PQDist_free(PQDist* pq);
void PQCaculate_Codes(PQDist* pq, float* vec, uint8_t* encode_vec);
uint8_t* extract_centroids_id(PQDist *pq, uint8_t* encode_data);
float* get_centroid_data(PQDist *pq, int quantizer, int code_id);
void load_query_data_and_cache(PQDist* pqdist, const float *_qdata);
float calc_dist_pq_loaded_by_id(PQDist* pqdist, uint8_t* ids);
//复制一份pqdist;

