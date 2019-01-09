#include <stdio.h>
#include <nvgraph.h>
#include "common.h"
#include "utils.h"
#define ceiling(a,b) ( (a+b-1)/b )

const char* version_name = "nvGraph";
int NUM, id;
int N, M;

typedef struct {
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphTraversalParameter_t traversal_param;
} nvGraph_t;

nvGraph_t* create_nvgraph(const dist_graph_t *graph, traverse_type_t traverse_type);

int *vis, *dis;
int *nowq, *nxtq, *lock, *turn;
int *v_pos, *e_dst;
float *e_weight, *ans, *nxtans;
void preprocess(dist_graph_t *graph, traverse_type_t traverse_type) {
    if(graph->p_num > 1) {
        printf("not implemented. Only support single-process.\n");
        fatal_error(11);
    }
    graph->additional_info = create_nvgraph(graph, traverse_type);
    N = graph->global_v;
    M = graph->global_e;
    cudaMalloc((void**)&v_pos, sizeof(int)*N );
    cudaMalloc((void**)&e_dst, sizeof(int)*M );
    cudaMalloc((void**)&e_weight, sizeof(float)*M );
    cudaMalloc((void**)&vis, sizeof(int)*N );
    cudaMalloc((void**)&dis, sizeof(int)*N );
    cudaMalloc((void**)&ans, sizeof(float)*N );
    cudaMalloc((void**)&nxtans, sizeof(float)*N );
    cudaMalloc((void**)&nowq, sizeof(int)*N );
    cudaMalloc((void**)&nxtq, sizeof(int)*N );
    cudaMalloc((void**)&lock, sizeof(int)*N );
    cudaMalloc((void**)&turn, sizeof(int)*N );
    cudaMemcpy( v_pos, graph->v_pos, sizeof(int)*N, cudaMemcpyHostToDevice );
    cudaMemcpy( e_dst, graph->e_dst, sizeof(int)*M, cudaMemcpyHostToDevice );
    cudaMemcpy( e_weight, graph->e_weight, sizeof(float)*M, cudaMemcpyHostToDevice );
}

nvGraph_t* create_nvgraph(const dist_graph_t *graph, traverse_type_t traverse_type) {
    struct nvgraphCSRTopology32I_st CSR_input;
    cudaDataType_t vertex_dimT[2] = {CUDA_R_32I, CUDA_R_32F};
    cudaDataType_t edge_dimT[1] = {CUDA_R_32F};
    nvGraph_t* nvgraph = (nvGraph_t*)malloc(sizeof(nvGraph_t));
    nvgraphCreate(&nvgraph->handle);
    nvgraphCreateGraphDescr(nvgraph->handle, &nvgraph->graph);
    CSR_input.nvertices = graph->global_v;
    CSR_input.nedges = graph->global_e;
    CSR_input.source_offsets = graph->v_pos;
    CSR_input.destination_indices = graph->e_dst;
    nvgraphSetGraphStructure(nvgraph->handle, nvgraph->graph, &CSR_input, NVGRAPH_CSR_32);
    if(traverse_type == BFS) {
        nvgraphAllocateVertexData(nvgraph->handle, nvgraph->graph, 1, vertex_dimT);
        nvgraphTraversalParameterInit(&nvgraph->traversal_param);
        nvgraphTraversalSetPredecessorsIndex(&nvgraph->traversal_param, 0);
        nvgraphTraversalSetUndirectedFlag(&nvgraph->traversal_param, true);
    } else {
        nvgraphAllocateVertexData(nvgraph->handle, nvgraph->graph, 2, vertex_dimT);
        nvgraphAllocateEdgeData  (nvgraph->handle, nvgraph->graph, 1, edge_dimT);
        nvgraphSetEdgeData(nvgraph->handle, nvgraph->graph, (void*)graph->e_weight, 0);
    }
    return nvgraph;
}

void destroy_additional_info(void *additional_info) {
    nvGraph_t* nvgraph = (nvGraph_t*)additional_info;
    nvgraphDestroyGraphDescr(nvgraph->handle, nvgraph->graph);
    nvgraphDestroy(nvgraph->handle);
    free(nvgraph);
    cudaFree(v_pos);
    cudaFree(e_dst);
}

__global__ void init( int *vis, int *dis, int N, int s )
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    if( x < N ){
        if( x == s ){
            vis[x] = x;
            dis[x] = 0;
        }
        else vis[x] = dis[x] = -1;
    }
}

__global__ void bfs_kernel( int *v_pos, int *e_dst, 
                             int *vis, int *dis, int cnt, 
                             int N, int *changed )
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int flag = 0;
    __shared__ int fl;
    if( threadIdx.x == 0 ) fl = 0;
    //__syncthreads();
    if( x < N ){
        if( dis[x] == cnt ){
            int begin = v_pos[x] - v_pos[0];
            int end = v_pos[x+1] - v_pos[0];
            for(int e = begin; e < end; ++e) {
                int v = e_dst[e];
                if( vis[v] == -1 ){
                    vis[v] = x;
                    dis[v] = cnt + 1;
                    flag = 1;
                }
            }   
        } 
    }
    if( flag ){
        fl = 1;
    }
    __syncthreads();
    if( threadIdx.x == 0 && fl ) *changed = 1;
}

void bfs(dist_graph_t *graph, index_t s, index_t* pred) { 
    if(graph->p_id == 0){
        nvGraph_t* nvgraph = (nvGraph_t*)graph->additional_info;
        int *changed, cnt = 0, tmp = 0, tt = 0;
        cudaMalloc((void **) &changed, sizeof(int));
        cudaMemcpy( changed, &tt, sizeof(int), cudaMemcpyHostToDevice );
        dim3 grid_size (ceiling(N,128));
        dim3 block_size (128);
        init<<<grid_size, block_size>>>( vis, dis, N, s );
        do{
            bfs_kernel<<<grid_size, block_size>>>(
                v_pos, e_dst, vis, dis, cnt, N, changed
            );
            cnt ++;
            cudaMemcpy( &tmp, changed, sizeof(int), cudaMemcpyDeviceToHost );
            //printf("ite %d %d\n", cnt, tmp);
            cudaMemcpy( changed, &tt, sizeof(int), cudaMemcpyHostToDevice );
        }while(tmp);

        cudaMemcpy(pred, vis, sizeof(int)*N, cudaMemcpyDeviceToHost);
        //printf("iteration %d %d root %d\n", cnt, s, pred[s]);
    }
}
/*
        nvgraphTraversal(nvgraph->handle, nvgraph->graph, NVGRAPH_TRAVERSAL_BFS, \
                &s, nvgraph->traversal_param);
        nvgraphGetVertexData(nvgraph->handle, nvgraph->graph, (void*)pred, 0);
        pred[s] = s;
*/

__global__ void sssp_init_ans( float *ans, float *nxtans, int *lock, int *turn, int N )
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    if( x < N ){
        nxtans[x] = ans[x];
        lock[x] = turn[x] = 0;
    }
}

__global__ void init_sssp( int *vis, float *ans, 
                           int *nowq, int *nxtq, int *lock,
                           int N, int s )
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    if( x < N ){
        if( x == s ){
            vis[x] = x;
            ans[x] = 0.0;
            nowq[x] = 1;
            nxtq[x] = 0;
        }
        else{
            vis[x] = -1;
            ans[x] = INFINITY;
            nowq[x] = nxtq[x] = 0;
        }
        lock[x] = 0;
    }
}

__global__ void sssp_kernel( int *v_pos, int *e_dst, float *e_weight,
                             int *vis, float *ans, float *nxtans,
                             int *nowq, int *nxtq, int *lock, int *turn,
                             int cnt, int N, int *changed )
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int flag = 0;
    __shared__ int fl;
    if( threadIdx.x == 0 ) fl = 0;
    //__syncthreads();
    if( x < N ){
        if( nowq[x] == cnt ){
            int begin = v_pos[x] - v_pos[0];
            int end = v_pos[x+1] - v_pos[0];
            for(int e = begin; e < end; ++e) {
                int v = e_dst[e];
                if( nxtans[v] > ans[x] + e_weight[e] ){
                    // lock
                    int myturn = atomicAdd( &lock[v], 1 );
                    while( turn[v] != myturn );

                    nxtans[v] = ans[x] + e_weight[e];
                    vis[v] = x;
                    nxtq[v] = cnt+1;
                    flag = 1;
                    
                    // unlock
                    turn[v] ++;
                }
            }   
        } 
    }
    if( flag ){
        fl = 1;
    }
    __syncthreads();
    if( threadIdx.x == 0 && fl ) *changed = 1;
}

void sssp(dist_graph_t *graph, index_t s, index_t* pred, weight_t* distance){
    if(graph->p_id == 0){
        int *sp = (int*)malloc( sizeof(int)*N );
        float *fp = (float*)malloc( sizeof(float)*N );
        nvGraph_t* nvgraph = (nvGraph_t*)graph->additional_info;
        int *changed, cnt = 0, tmp = 0, tt = 0;
        cudaMalloc((void **) &changed, sizeof(int));
        cudaMemcpy( changed, &tt, sizeof(int), cudaMemcpyHostToDevice );
        dim3 grid_size (ceiling(N,128));
        dim3 block_size (128);
        for( int i = 0; i < M; i ++ ){
            if( graph->e_weight[i] < 0.0 ) printf("%d %f\n", i, graph->e_weight[i] );
        }
        init_sssp<<<grid_size, block_size>>>( vis, ans, nowq, nxtq, lock, N, s );
        do{
            cnt ++;
            sssp_init_ans<<<grid_size, block_size>>>( ans, nxtans, lock, turn, N );
            printf("ite %d ", cnt);
            cudaMemcpy( sp, nowq, sizeof(int)*N, cudaMemcpyDeviceToHost );
            int num = 0;
            for( int i = 0; i < N; i ++ ){
                if( sp[i] == cnt ){
                    printf("%d ", i);
                    num ++;
                }
            }
            printf("\nnum %d\n", num);
            
            printf("ans\n");
            num = 0;
            cudaMemcpy( fp, ans, sizeof(float)*N, cudaMemcpyDeviceToHost );
            for( int i = 0; i < N; i ++ ){
                if( fp[i] != INFINITY ){
                    printf("%d : %f\n", i, fp[i]);
                    num ++;
                }
            }
            printf("num %d\n", num);

            // printf("nxtans\n");
            // num = 0;
            // cudaMemcpy( fp, nxtans, sizeof(float)*N, cudaMemcpyDeviceToHost );
            // for( int i = 0; i < N; i ++ ){
            //     if( fp[i] != INFINITY ){
            //         printf("%d : %f\n", i, fp[i]);
            //         num ++;
            //     }
            // }
            // printf("num %d\n", num);

            sssp_kernel<<<grid_size, block_size>>>(
                v_pos, e_dst, e_weight, vis, ans, nxtans,
                nowq, nxtq, lock, turn, cnt, N, changed
            );
            // int debug[10];
            // cudaMemcpy( debug, lock, sizeof(int)*2, cudaMemcpyDeviceToHost );
            // printf("d0 %d d1 %d\n", debug[0], debug[1]);

            cudaMemcpy( &tmp, changed, sizeof(int), cudaMemcpyDeviceToHost );
            printf("ite %d %d\n", cnt, tmp);
            cudaMemcpy( changed, &tt, sizeof(int), cudaMemcpyHostToDevice );
            int *tq = nowq;
            nowq = nxtq;
            nxtq = tq;
            float *tp = ans;
            ans = nxtans;
            nxtans = tp;
            if( cnt >= 20 ) break;
        }while(tmp);

        cudaMemcpy(pred, vis, sizeof(int)*N, cudaMemcpyDeviceToHost);
        cudaMemcpy(distance, ans, sizeof(float)*N, cudaMemcpyDeviceToHost);
        printf("iteration %d\n", cnt);
    }
}