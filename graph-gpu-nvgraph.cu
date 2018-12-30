#include <stdio.h>
#include <nvgraph.h>
#include "common.h"
#include "utils.h"

const char* version_name = "nvGraph";

typedef struct {
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphTraversalParameter_t traversal_param;
} nvGraph_t;

nvGraph_t* create_nvgraph(const dist_graph_t *graph, traverse_type_t traverse_type);

void preprocess(dist_graph_t *graph, traverse_type_t traverse_type) {
    if(graph->p_num > 1) {
        printf("not implemented. Only support single-process.\n");
        fatal_error(11);
    }
    graph->additional_info = create_nvgraph(graph, traverse_type);
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
}

void bfs(dist_graph_t *graph, index_t s, index_t* pred){
    if(graph->p_id == 0){
        nvGraph_t* nvgraph = (nvGraph_t*)graph->additional_info;
        nvgraphTraversal(nvgraph->handle, nvgraph->graph, NVGRAPH_TRAVERSAL_BFS, \
                &s, nvgraph->traversal_param);
        nvgraphGetVertexData(nvgraph->handle, nvgraph->graph, (void*)pred, 0);
        pred[s] = s;
    }
}

void sssp(dist_graph_t *graph, index_t s, index_t* pred, weight_t* distance){
    if(graph->p_id == 0){
        nvGraph_t* nvgraph = (nvGraph_t*)graph->additional_info;
        nvgraphSssp(nvgraph->handle, nvgraph->graph, 0,  &s, 0);
        nvgraphGetVertexData(nvgraph->handle, nvgraph->graph, (void*)distance, 0);
        printf("not implemented. nvgraphSssp() does not calculate the predecessors\n");
    }
    fatal_error(11);
}