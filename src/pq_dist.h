#pragma once
#include <stdlib.h>
#include <stdint.h>

typedef struct{
    int d;
    int m;
    int nbits;
    int code_nums;
    int d_pq;
    size_t table_size;
    uint8_t *codes;
    float* centroids;
    float* pq_dist_cache_data;

}PQDist;

void pqdist_init(PQDist* pqdist, int _d, int _m, int _nbits);

void PQDist_load(PQDist* pq, const char* filename);
void PQDist_free(PQDist* pq);