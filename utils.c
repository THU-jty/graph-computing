#define _GNU_SOURCE
#include <float.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "common.h"
#include "utils.h"

#define CHECK_AND_SET(cond, state) if(cond) {state;}
#define CHECK_AND_BREAK(cond, state) if(cond) {state;break;}

int my_file_read_at_all(MPI_File fh, MPI_Offset offset, void *buf,\
                        int count, MPI_Datatype datatype);
int read_graph_impl(dist_graph_t *graph, MPI_File file, \
                  int global_v, int local_v, int offset_v, traverse_type_t type);
int read_graph_with_distribution(dist_graph_t *graph, const char* filename, \
                    int local_v, int offset_v, traverse_type_t type);
/*
typedef struct {
    int global_v, local_v, offset_v;
    const int* v_count;
    const int* v_displs;
} tree_info_t;*/

void coo2csr(int nv, int ne, index_t* row_idx, index_t* col_idx,  index_t* row_pos);

/* check traverse tree */
int check_tree(index_t global_v, index_t s, \
                 const index_t* pred, index_t* level);

/* check whether all edges relaxed */
int check_bfs_edges(const dist_graph_t *graph, index_t s, \
                  const index_t* pred, const index_t* level);
int check_sssp_edges(const dist_graph_t *graph, index_t s, \
                  const index_t* pred, const weight_t* distance);

int read_graph_default(dist_graph_t *graph, const char* filename, traverse_type_t type) {
    MPI_File file;
    int global_v, global_e;
    int local_v, offset_v, r, q;
    index_t* v_pos;
    index_t *e_dst = NULL;
    weight_t *e_weight = NULL;
    int ret = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    CHECK_ERROR(ret, ret)
    ret = my_file_read_at_all(file, 0, &global_v, 1, MPI_INT);
    CHECK_ERROR(ret, ret)

    q = global_v / graph->p_num;
    r = global_v % graph->p_num;
    local_v = q + ((graph->p_id < r) ? 1 : 0);
    offset_v = graph->p_id * q + ((graph->p_id < r) ? graph->p_id : r);

    ret = read_graph_impl(graph, file, global_v, local_v, offset_v, type);
    MPI_File_close(&file);
    return ret;
}

int read_graph_with_distribution(dist_graph_t *graph, const char* filename, \
                    int local_v, int offset_v, traverse_type_t type) {
    MPI_File file;
    int global_v, global_e;
    index_t* v_pos;
    index_t *e_dst = NULL;
    weight_t *e_weight = NULL;
    int ret = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    CHECK_ERROR(ret, ret)
    ret = my_file_read_at_all(file, 0, &global_v, 1, MPI_INT);
    CHECK_ERROR(ret, ret)

    ret = read_graph_impl(graph, file, global_v, local_v, offset_v, type);
    MPI_File_close(&file);
    return ret;
}

int read_graph_impl(dist_graph_t *graph, MPI_File file, \
                  int global_v, int local_v, int offset_v, traverse_type_t type) {
    MPI_Offset file_offset;
    int global_e, local_e;
    index_t* v_pos = NULL;
    index_t *e_dst = NULL;
    weight_t *e_weight = NULL;
    int ret;

    v_pos = (index_t*)malloc(sizeof(index_t) * (local_v + 1));
    CHECK(v_pos == NULL, MPI_ERR_NO_MEM)

    file_offset = (offset_v + 1) * sizeof(int);
    ret = my_file_read_at_all(file, file_offset, v_pos, local_v + 1, MPI_INT);
    CHECK_ERROR(ret, ret)
    global_e = v_pos[local_v];
    MPI_Bcast(&global_e, 1, MPI_INT, graph->p_num-1, MPI_COMM_WORLD);
    
    local_e = v_pos[local_v] - v_pos[0];
    if(local_e > 0) {
        e_dst = (index_t*) malloc(sizeof(index_t) * local_e);
        CHECK_NULL(e_dst, MPI_ERR_NO_MEM)
        if(type == SSSP) {
            e_weight = (weight_t*) malloc(sizeof(weight_t) * local_e);
            CHECK_NULL(e_weight, MPI_ERR_NO_MEM)
        }
    }

    file_offset = (global_v + 2 + v_pos[0]) * sizeof(int);
    ret = my_file_read_at_all(file, file_offset, e_dst, local_e, MPI_INT);
    CHECK_ERROR(ret, ret)
    if(type == SSSP) {
        file_offset = (global_v + 2 + global_e) * sizeof(int) + v_pos[0] * sizeof(float);
        ret = my_file_read_at_all(file,  file_offset, e_weight, local_e, MPI_FLOAT);
        CHECK_ERROR(ret, ret)
    }

    graph->global_v = global_v;
    graph->global_e = global_e;
    graph->local_v = local_v;
    graph->offset_v = offset_v;
    graph->local_e = local_e;
    graph->v_pos = v_pos;
    graph->e_dst = e_dst;
    graph->e_weight = e_weight;
    graph->additional_info = NULL;
    graph->gpu_v_pos = NULL;
    graph->gpu_e_dst = NULL;
    graph->gpu_e_weight = NULL;
    graph->on_gpu = false;
    graph->twoD_partitioned = false;
    graph->cpu_free = free;
    graph->gpu_free = NULL;
    return MPI_SUCCESS;
}

int my_file_read_at_all(MPI_File fh, MPI_Offset offset, void *buf,\
                        int count, MPI_Datatype datatype) {
    int actual_count;
    MPI_Status status;
    int ret = MPI_File_read_at_all(fh, offset, buf, count, datatype, &status);
    CHECK_ERROR(ret, ret)
    ret = MPI_Get_count(&status, datatype, &actual_count);
    CHECK(count != actual_count, MPI_ERR_IO)
    return ret;
}


int check_answer(dist_graph_t *graph, const char* filename, index_t s, \
                 traverse_type_t type, const index_t* pred, const weight_t* distance) {  
    //tree_info_t tree;           
    int err = 0, err_local = 0;
    int offset_v = graph->offset_v, local_v = graph->local_v;
    int global_v = graph->global_v, p_num = graph->p_num;
    index_t* global_pred = NULL;
    weight_t* global_distance = NULL;
    index_t* v_count = NULL;
    index_t* v_displs = NULL;
    index_t *level;

    /* reload graph */
    destroy_dist_graph(graph);
    err = read_graph_with_distribution(graph, filename, local_v, offset_v, type);
    CHECK_ERROR(err, err)
    v_count = (index_t*) malloc(sizeof(index_t) * p_num * 2);
    v_displs = v_count + p_num;
    CHECK_NULL(v_count, MPI_ERR_NO_MEM)

    level = (index_t*)malloc(sizeof(index_t) * global_v * 2);
    global_pred = level + global_v;
    CHECK_NULL(level, MPI_ERR_NO_MEM)

    if(type == SSSP) {
        global_distance = (weight_t*)malloc(sizeof(weight_t) * global_v);
        CHECK_NULL(global_distance, MPI_ERR_NO_MEM)
    }
    
    /* duplicate the answer over all processes */
    MPI_Allgather(&local_v, 1, MPI_INT, v_count, 1, MPI_INT,  MPI_COMM_WORLD);
    MPI_Allgather(&offset_v, 1, MPI_INT, v_displs, 1, MPI_INT, MPI_COMM_WORLD);
    
    {
        int prev = 0;
        for(int i = 0; i < p_num; ++i) {
            CHECK_ERROR(v_displs[i] != prev || v_count[i] < 0, 1)
            prev += v_count[i];
        }
        CHECK_ERROR(prev != global_v, 1)
    }

    /* duplicate the answer over all processes */
    MPI_Allgatherv(pred, local_v, MPI_INT,
            global_pred, v_count, v_displs, MPI_INT, MPI_COMM_WORLD);
    if(type == SSSP) {
        MPI_Allgatherv(distance, local_v, MPI_FLOAT,
                global_distance, v_count, v_displs, MPI_FLOAT, MPI_COMM_WORLD);
    }
/*
    tree.global_v = global_v;
    tree.local_v = local_v;
    tree.offset_v = offset_v;
    tree.v_count = v_count;
    tree.v_displs = v_displs;*/
    if(graph->p_id == 0) {
        err = check_tree(global_v, s, global_pred, level);
        CHECK_ERROR(err, err)
    }
    MPI_Bcast(level, global_v, MPI_INT, 0, MPI_COMM_WORLD);

    if(type == BFS) {
        err_local = check_bfs_edges(graph, s, global_pred, level);
    } else {
        err_local = check_sssp_edges(graph, s, global_pred, global_distance);
        free(global_distance);
    }
    free(v_count);
    free(level);
    MPI_Reduce(&err_local, &err, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    return err;
}


int check_tree(index_t global_v, index_t s, \
                 const index_t* pred, index_t* level) {
    typedef index_t *ptr_t;
    int complete, unreachable, reachable;
    int err, begin, end;
    ptr_t succ, v_pos, queue, tmp_buffer;
    
    unreachable = 0;
    for(int i = 0; i < global_v; ++i) {
        CHECK(pred[i] >= global_v, 2)
        if(pred[i] < 0) {
            ++unreachable;
        } else {        
            CHECK(pred[pred[i]] < 0, 3)
        }
    }
    CHECK(pred[s] != s, 4)
    
    v_pos = (index_t*)malloc(sizeof(index_t) * (global_v * 3 + 1 - unreachable));
    succ = v_pos + global_v + 1;
    tmp_buffer = succ + global_v - unreachable;
    CHECK_NULL(v_pos, MPI_ERR_NO_MEM)
    
    reachable = 0;
    for(int i = 0; i < global_v; ++i) {
        level[i] = -1;
        if(pred[i] >= 0 && i != s) {
            succ[reachable] = i;
            tmp_buffer[reachable] = pred[i];
            reachable++;
        }
    }
    coo2csr(global_v, reachable, tmp_buffer, succ, v_pos);

    queue = tmp_buffer;
    level[s] = 0;
    begin = 0;
    end = 1;
    queue[0] = s;
    
    while(end > begin) {  /* perform bfs on the tree */
        int u = queue[begin++];
        for(int e = v_pos[u]; e < v_pos[u+1]; ++e) {
            int v = succ[e];
            level[v] = level[u] + 1;
            queue[end++] = v;
        }
    }
    free(v_pos);
    CHECK(end + unreachable != global_v, 5) /* has loop in the tree */
    return 0;
}

/*
int check_tree(const tree_info_t* tree, index_t s, \
                 const index_t* pred, index_t* level) {
    const index_t* v_count = tree->v_count;
    const index_t* v_displs = tree->v_displs;
    int global_v = tree->global_v;
    int local_v = tree->local_v;
    int offset_v = tree->offset_v;
    int complete, unreachable_local = 0, unreachable;
    int err_local = 0, err, size_f_local, size_f;
    for(int i = offset_v; i < local_v + offset_v; ++i) {
        CHECK_AND_SET(pred[i] >= global_v, err_local = 2)
        if(pred[i] < 0) {
            ++unreachable_local;
        }
        level[i] = -1;
    }
    MPI_Allreduce(&err_local, &err, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    CHECK(err, err)
    CHECK_AND_SET(pred[s] != s, err_local = 3)
    MPI_Allreduce(&err_local, &err, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    CHECK(err, err)
    MPI_Allgatherv(MPI_IN_PLACE, local_v, MPI_INT,
                   level, v_count, v_displs,  MPI_INT, MPI_COMM_WORLD);
    MPI_Allreduce(&unreachable_local, &unreachable, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    level[s] = 0;
    size_f = 1;
    complete = unreachable + 1;
    for(int current_level = 0; size_f > 0 && complete < global_v; ++ current_level) {
        size_f_local = 0;
        for(int i = offset_v; i < local_v + offset_v; ++i) {
            if(level[i] == -1 && pred[i] >= 0 && level[pred[i]] == current_level) {
                level[i] = current_level + 1;
                ++ size_f_local;
            }
        }
        MPI_Allgatherv(MPI_IN_PLACE, local_v, MPI_INT,
                       level, v_count, v_displs,  MPI_INT, MPI_COMM_WORLD);
        MPI_Allreduce(&size_f_local, &size_f, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        complete += size_f;
    }
    CHECK(complete != global_v, 4)
    return 0;
}
*/
int check_bfs_edges(const dist_graph_t *graph, index_t s, \
                  const index_t* pred, const index_t* level) {
    const int local_v = graph->local_v;
    const int offset_v = graph->offset_v;
    const index_t *v_pos = graph->v_pos;
    const index_t *e_dst = graph->e_dst;
    const weight_t *e_weight = graph->e_weight;
    int ret = 0;

#ifdef _OPENMP
#pragma omp parallel for shared(ret)
#endif
    for(int idx = 0; idx < local_v; ++idx) {
        int err = 0;
        int i = offset_v + idx;
        bool i_reached = pred[i] >= 0;
        int begin = v_pos[idx] - v_pos[0];
        int end = v_pos[idx+1] - v_pos[0];
        for(int e = begin; e < end && e_dst[e] < i; ++e) {
            int j = e_dst[e];
            bool j_reached = pred[j] >= 0;
            CHECK_AND_BREAK(i_reached != j_reached, err = 6)
            if(i_reached && j_reached) {
                CHECK_AND_BREAK(abs(level[i] - level[j]) > 1, err = 7)
            }
        }
        if(err != 0) {
#ifdef _OPENMP
#pragma omp atomic write
#endif
            ret = err;
        }
    }
    return ret;
}

weight_t max_f(weight_t a, weight_t b) {
    return (a > b) ? a : b;
}

int check_sssp_edges(const dist_graph_t *graph, index_t s, \
                  const index_t* pred, const weight_t* distance) {
    const int local_v = graph->local_v;
    const int offset_v = graph->offset_v;
    const index_t *v_pos = graph->v_pos;
    const index_t *e_dst = graph->e_dst;
    const weight_t *e_weight = graph->e_weight;
    int ret = 0;
#ifdef _OPENMP
#pragma omp parallel for shared(ret)
#endif
    for(int idx = 0; idx < local_v; ++idx) {
        int err = 0;
        int i = offset_v + idx;
        bool i_reached = pred[i] >= 0;
        int begin = v_pos[idx] - v_pos[0];
        int end = v_pos[idx+1] - v_pos[0];
        for(int e = begin; e < end && e_dst[e] < i; ++e) {
            int j = e_dst[e];
            bool j_reached = pred[j] >= 0;
            CHECK_AND_BREAK(i_reached != j_reached, err = 6)
            if(i_reached && j_reached) {
                if(distance[i] > distance[j]) {
                    weight_t eps = FLT_EPSILON * max_f(distance[i], (distance[j] + e_weight[e]));
                    weight_t diff = distance[i] - (distance[j] + e_weight[e]);
                    if(pred[i] == j) {
                        CHECK_AND_BREAK(fabsf(diff) > eps, err = 8)
                    } else {
                        CHECK_AND_BREAK(diff > eps, err = 7)
                    }
                } else {
                    weight_t eps = FLT_EPSILON * max_f(distance[j], (distance[i] + e_weight[e]));
                    weight_t diff = distance[j] - (distance[i] + e_weight[e]);
                    if(pred[j] == i) {
                        CHECK_AND_BREAK(fabsf(diff) > eps, err = 8)
                    } else {
                        CHECK_AND_BREAK(diff > eps, err = 7)
                    }
                }
            }
        }
        if(err != 0) {
#ifdef _OPENMP
#pragma omp atomic write
#endif
            ret = err;
        }
    }
    return ret;
}

int get_te(const dist_graph_t *graph, const index_t* pred) {
    int te_local = 0, te = 0;
    for(int i = 0; i < graph->local_v; ++i) {
        if(pred[i] >= 0) {
            te_local += (graph->v_pos[i+1] - graph->v_pos[i]);
        }
    }
    MPI_Allreduce(&te_local, &te, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return te;
}

int fatal_error(int code) {
    return MPI_Abort(MPI_COMM_WORLD, code);
}

void destroy_dist_graph(dist_graph_t *graph) {
    if(graph->additional_info != NULL){
        destroy_additional_info(graph->additional_info);
        graph->additional_info = NULL;
    }
    if(graph->cpu_free != NULL) {
        if(graph->v_pos != NULL){
            graph->cpu_free(graph->v_pos);
            graph->v_pos = NULL;
        }
        if(graph->e_dst != NULL){
            graph->cpu_free(graph->e_dst);
            graph->e_dst = NULL;
        }
        if(graph->e_weight != NULL){
            graph->cpu_free(graph->e_weight);
            graph->e_weight = NULL;
        }
    }
    if(graph->cpu_free != NULL) {
        if(graph->gpu_v_pos != NULL){
            graph->gpu_free(graph->gpu_v_pos);
            graph->gpu_v_pos = NULL;
        }
        if(graph->gpu_e_dst != NULL){
            graph->gpu_free(graph->gpu_e_dst);
            graph->gpu_e_dst = NULL;
        }
        if(graph->gpu_e_weight != NULL){
            graph->gpu_free(graph->gpu_e_weight);
            graph->gpu_e_weight = NULL;
        }
    }
}

#define RADIX 256
#define MASK 255
#define BITWIDTH 8

void radix_sort(int max_v, int size, index_t* key, index_t* attach, index_t* buffer) {
    for(int shift = 0; max_v > 0; shift += BITWIDTH, max_v = max_v >> BITWIDTH) {
        int count[RADIX], begin[RADIX], prefix;

        memset(count, 0, sizeof(int) * RADIX);    
        for(int i = 0; i < size; ++i) {
            int bin = (key[i] >> shift) & MASK;
            count[bin] ++;
        }
        prefix = 0;
        for(int bin = 0; bin < RADIX; ++ bin) {
            int c = count[bin];
            count[bin] = prefix;
            prefix += c;
        }
        memcpy(begin, count, sizeof(int) * RADIX);
        for(int i = 0; i < size; ++i) {
            int bin = (key[i] >> shift) & MASK;
            buffer[begin[bin]++] = attach[i];
        }
        memcpy(attach, buffer, sizeof(index_t) * size);
        memcpy(begin, count, sizeof(int) * RADIX);
        for(int i = 0; i < size; ++i) {
            int bin = (key[i] >> shift) & MASK;
            buffer[begin[bin]++] = key[i];
        }
        memcpy(key, buffer, sizeof(index_t) * size);
    }
}

void coo2csr(int nv, int ne, index_t* row_idx, index_t* col_idx,  index_t* row_pos) {
    radix_sort(nv, ne, row_idx, col_idx, row_pos); 
    for(int i = 0; i <= row_idx[0]; ++i) {
        row_pos[i] = 0;
    }
    for(int e = 1; e < ne; ++e) {
        for(int i = row_idx[e - 1] + 1; i <= row_idx[e]; ++i) {
            row_pos[i] = e;
        }
    }
    for(int i = row_idx[ne - 1] + 1; i <= nv; ++i) {
        row_pos[i] = ne;
    }
}
