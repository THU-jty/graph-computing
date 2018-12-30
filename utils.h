#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED 1

#include "common.h"

#define CHECK(err, err_code) if(err) { return err_code; }
#define CHECK_ERROR(ret, err_code) CHECK(ret != 0, err_code)
#define CHECK_NULL(ret, err_code) CHECK(ret == NULL, err_code)

#ifdef __cplusplus
extern "C" {
#endif

int fatal_error(int code);

int read_graph_default(dist_graph_t *graph, const char* filename, traverse_type_t type);
void destroy_dist_graph(dist_graph_t *graph);

int get_te(const dist_graph_t *graph, const index_t* pred);
int check_answer(dist_graph_t *graph, const char* filename, index_t s, \
        traverse_type_t type, const index_t* pred, const weight_t* distance);

#ifdef __cplusplus
}
#endif

#endif