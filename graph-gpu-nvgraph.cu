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
                             int *vis, int *dis,
                             int *nowq, int *nxtq, 
                             int now_size, int *nxt_size )
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    if( x < now_size ){
        int u = nowq[x];
        int begin = v_pos[u] - v_pos[0];
        int end = v_pos[u+1] - v_pos[0];
        for(int e = begin; e < end; ++e) {
            int v = e_dst[e];
            if( atomicCAS( &vis[v], -1, u ) == -1 ){
                int position = atomicAdd(nxt_size, 1);
                nxtq[position] = v;
            }
        }   
    }
}

int *nowq, *nxtq;
void bfs(dist_graph_t *graph, index_t s, index_t* pred) { 
    if(graph->p_id == 0){
        nvGraph_t* nvgraph = (nvGraph_t*)graph->additional_info;
        cudaMalloc((void**)&vis, sizeof(int)*N );
        cudaMalloc((void**)&dis, sizeof(int)*N );
        cudaMalloc((void**)&nowq, sizeof(int)*N );
        cudaMalloc((void**)&nxtq, sizeof(int)*N );
        int *size, cnt = 0, tmp = s, tt = 1, ite = 0;
        cudaMalloc((void **) &size, sizeof(int));
        cudaMemcpy( size, &cnt, sizeof(int), cudaMemcpyHostToDevice );
        cudaMemcpy( nowq, &tmp, sizeof(int), cudaMemcpyHostToDevice );

        cudaMemcpy( &tmp, nowq, sizeof(int), cudaMemcpyDeviceToHost );

        dim3 grid_size (ceiling(N,1024));
        dim3 block_size (1024);
        init<<<grid_size, block_size>>>( vis, dis, N, s );
        tmp = 1; tt = 0;
        int st = 0;
        //printf("queue in size %d\t", tmp);
        do{
            ite ++;
            bfs_kernel<<<grid_size, block_size>>>(
                v_pos, e_dst, vis, dis,
                nowq, nxtq, tmp, size
            );
            cudaMemcpy( &tmp, size, sizeof(int), cudaMemcpyDeviceToHost );
            //printf("%d queue out size %d\n", ite, tmp);
            cudaMemcpy( size, &tt, sizeof(int), cudaMemcpyHostToDevice );
            int *tq = nowq;
            nowq = nxtq;
            nxtq = tq;
        }while(tmp);

        cudaMemcpy(pred, vis, sizeof(int)*N, cudaMemcpyDeviceToHost);
        cudaFree(vis);
        cudaFree(dis);
        cudaFree(nowq);
        cudaFree(nxtq);
        //printf("iteration %d\n", ite);
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