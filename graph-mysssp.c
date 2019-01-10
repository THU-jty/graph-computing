#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include "common.h"

const char* version_name = "A DIV sssp base-line";\

int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}
int *vis, *rev;

/* naive implementation uses only process 0 */
void preprocess(dist_graph_t *graph, traverse_type_t traverse_type) {
    /* centralize the graph data */
    int offset_v, local_v = 0, local_e = 0;
    index_t* v_pos = NULL;
    index_t* e_dst = NULL;
    weight_t* e_weight = NULL;
    index_t* v_count;
    index_t* v_displs;
    index_t* e_count;
    index_t* e_displs;

    if(graph->p_id == 0) {
        v_count = (index_t*) malloc(graph->p_num * 4 * sizeof(index_t));
        v_displs = v_count + graph->p_num;
        e_count = v_count + graph->p_num * 2;
        e_displs = v_count + graph->p_num * 3;
        local_v = graph->global_v;
        local_e = graph->global_e;
        offset_v = 0;
        v_pos = (index_t*)malloc(sizeof(index_t) * (graph->global_v+1));
        e_dst = (index_t*)malloc(sizeof(index_t) * graph->global_e);
        if(traverse_type == SSSP) {
            e_weight = (weight_t*)malloc(sizeof(weight_t) * graph->global_e);
        }
    } else {
        offset_v = graph->global_v;
    }
    
    /* size first */
    MPI_Gather(&graph->local_v, 1, MPI_INT, \
            v_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&graph->local_e, 1, MPI_INT, \
            e_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(graph->p_id == 0) {
        int v_prev = 0, e_prev = 0;
        for(int i = 0; i < graph->p_num; ++i) {
            v_displs[i] = v_prev;
            e_displs[i] = e_prev;
            v_prev += v_count[i];
            e_prev += e_count[i];
        }
    }
    
    /* data second */
    MPI_Gatherv(graph->v_pos, graph->local_v, MPI_INT,
            v_pos, v_count, v_displs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(graph->e_dst, graph->local_e, MPI_INT,
            e_dst, e_count, e_displs, MPI_INT, 0, MPI_COMM_WORLD);

    free(graph->v_pos);
    graph->v_pos = v_pos;
    free(graph->e_dst);
    graph->e_dst = e_dst;
    if(traverse_type == SSSP) {
        MPI_Gatherv(graph->e_weight, graph->local_e, MPI_FLOAT,
            e_weight, e_count, e_displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
        free(graph->e_weight);
        graph->e_weight = e_weight;
    }
    
    if(graph->p_id == 0) {
        v_pos[local_v] = local_e;
        free(v_count);
        graph->additional_info = malloc(sizeof(index_t) * graph->global_v * 2);
    }
    vis = (int*)malloc( sizeof(int)*graph->global_v );
    rev = (int*)malloc( sizeof(int)*graph->global_v );
    graph->local_v = local_v;
    graph->offset_v = offset_v;
    graph->local_e = local_e;
}

void destroy_additional_info(void *additional_info) {
    free(additional_info);
}

/* sequential BFS with queue */
void bfs(dist_graph_t *graph, index_t s, index_t* pred){
    if(graph->p_id == 0){
        const index_t* v_pos = graph->v_pos;
        const index_t* e_dst = graph->e_dst;
        const int offset_v = graph->offset_v, local_v = graph->local_v;
        const int global_v = graph->global_v;
        index_t* queue_f = (index_t*)graph->additional_info;
        index_t* queue_g = queue_f + global_v;
        int size_f = 0;

        for(int i = 0; i < local_v; ++i) {
            pred[i] = UNREACHABLE;
        }
        pred[s] = s;
        queue_f[size_f++] = s;
        do {
            index_t* tmp;
            int size_g = 0;
            for(int i = 0; i < size_f; ++i){
                int u = queue_f[i];
                int begin = v_pos[u] - v_pos[0];
                int end = v_pos[u + 1] - v_pos[0];
                for(int e = begin; e < end; ++e) {
                    int v = e_dst[e];
                    if(pred[v] == UNREACHABLE) {
                        pred[v] = u;
                        queue_g[size_g++] = v;
                    }
                }
            }
            size_f = size_g;
            tmp = queue_f;
            queue_f = queue_g;
            queue_g = tmp;
        } while(size_f > 0);
    }
}

float *a, *b;
/* sequential SPFA with bitmap */
void sssp(dist_graph_t *graph, index_t s, index_t* pred, weight_t* distance){
    if(graph->p_id == 0){
        const index_t* v_pos = graph->v_pos;
        const index_t* e_dst = graph->e_dst;
        const weight_t* e_weight = graph->e_weight;
        const int offset_v = graph->offset_v, local_v = graph->local_v;
        const int global_v = graph->global_v;
        int* queue_f = (int*)graph->additional_info;
        int* queue_g = queue_f + global_v;
        int size_f = 0, size_g = 0;
        int N = global_v;
        b = (float*)malloc(sizeof(float)*N);
        for(int i = 0; i < local_v; ++i) {
            distance[i] = INFINITY;
            b[i] = INFINITY;
            pred[i] = UNREACHABLE;
            vis[i] = 0;
        }
        pred[s] = s;
        distance[s] = b[s] = 0.0f;
        queue_f[ size_f++ ] = s;
        a = distance;

        int it = 0;
        do {
            it ++;
            int* tmp;
            size_g = 0;
            //printf("it %d num %d\n", it, size_f);
            for( int i = 0; i < N; i ++ ) b[i] = a[i];
            for( int i = 0; i < size_f; i ++ ){
                int u = queue_f[i];
                int begin = v_pos[u] - v_pos[0];
                int end = v_pos[u + 1] - v_pos[0];
                for(int e = begin; e < end; ++e) {
                    int v = e_dst[e];
                    if(b[v] > a[u] + e_weight[e]) {
                        if( v == 68 && it == 2 ){
                            //printf("before %f after %f src %d dis %f wgt %f\n", b[v], a[u]+e_weight[e], u, a[u], e_weight[e]);
                        }
                        b[v] = a[u] + e_weight[e];
                        pred[v] = u;
                        if( vis[v] != it ){
                            vis[v] = it;
                            queue_g[ size_g++ ] = v;
                        }
                    }
                    
                }
            }

            tmp = queue_f;
            queue_f = queue_g;
            queue_g = tmp;
            int tt = size_f;
            size_f = size_g;
            size_g = tt;
            for( int i = 0; i < N; i ++ ) printf("%d %6d %.6f\n", it, i, b[i]);

            float *tsp = b;
            b = a;
            a = tsp;
        } while(size_f > 0);
        puts("");
    }
}
