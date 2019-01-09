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
int *nowq, *nxtq;
int *v_pos, *e_dst;
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
    cudaMalloc((void**)&vis, sizeof(int)*N );
    cudaMalloc((void**)&dis, sizeof(int)*N );
    cudaMalloc((void**)&nowq, sizeof(int)*N );
    cudaMalloc((void**)&nxtq, sizeof(int)*N );
    cudaMemcpy( v_pos, graph->v_pos, sizeof(int)*N, cudaMemcpyHostToDevice );
    cudaMemcpy( e_dst, graph->e_dst, sizeof(int)*M, cudaMemcpyHostToDevice );
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
        cudaMalloc((void**)&vis, sizeof(int)*N );
        cudaMalloc((void**)&dis, sizeof(int)*N );
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
            printf("ite %d %d\n", cnt, tmp);
            cudaMemcpy( changed, &tt, sizeof(int), cudaMemcpyHostToDevice );
        }while(tmp);

        cudaMemcpy(pred, vis, sizeof(int)*N, cudaMemcpyDeviceToHost);
        cudaFree(vis);
        cudaFree(dis);
        printf("iteration %d\n", cnt);
    }
}
/*
        nvgraphTraversal(nvgraph->handle, nvgraph->graph, NVGRAPH_TRAVERSAL_BFS, \
                &s, nvgraph->traversal_param);
        nvgraphGetVertexData(nvgraph->handle, nvgraph->graph, (void*)pred, 0);
        pred[s] = s;
*/

void sssp(dist_graph_t *graph, index_t s, index_t* pred, weight_t* distance){
    if(graph->p_id == 0){
        nvGraph_t* nvgraph = (nvGraph_t*)graph->additional_info;
        nvgraphSssp(nvgraph->handle, nvgraph->graph, 0,  &s, 0);
        nvgraphGetVertexData(nvgraph->handle, nvgraph->graph, (void*)distance, 0);
        printf("not implemented. nvgraphSssp() does not calculate the predecessors\n");
    }
    fatal_error(11);
}