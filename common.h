#ifndef COMMON_H_INCLUDED
#define COMMON_H_INCLUDED 1

#include<stdint.h>
#include<stdbool.h>

#define UNREACHABLE (-1)
typedef int32_t index_t;
typedef float weight_t;
typedef void (*free_func_t)(void*);

/* 
 * global_v: number of vertices in the whole input graph
 * global_e: number of edges in the whole input graph
 * local_v: number of vertices in the current process
 * offset_v: number of vertices in previous processes
 * local_e: number of edges in the current process
 */
typedef struct {
    int p_id, p_num;                 /* do not modify */
    int global_v, global_e;          /* do not modify */
    int local_v, offset_v, local_e;  /* do not modify after preprocess */
    
    index_t* v_pos;
    index_t* e_dst;
    weight_t* e_weight;
    free_func_t cpu_free;

    bool twoD_partitioned;           /* if 2D partitioned */
    int local_v_src, local_v_dst;    /* may be used for 2D partitioning */
    int offset_v_src, offset_v_dst;  /* may be used for 2D partitioning */

    bool on_gpu;                     /* if GPU is used */
    free_func_t gpu_free;
    index_t* gpu_v_pos;              /* may be used for GPU */
    index_t* gpu_e_dst;              /* may be used for GPU */
    weight_t* gpu_e_weight;          /* may be used for GPU */

    void *additional_info;           /* any information you want to attach */
} dist_graph_t;

typedef enum {
    BFS, SSSP
} traverse_type_t;

#ifdef __cplusplus
extern "C" {
#endif

void preprocess(dist_graph_t *graph, traverse_type_t traverse_type);
void destroy_additional_info(void *additional_info);
void bfs(dist_graph_t *graph, index_t s, index_t* pred);
void sssp(dist_graph_t *graph, index_t s, index_t* pred, weight_t* distance);

#ifdef __cplusplus
}
#endif

#endif