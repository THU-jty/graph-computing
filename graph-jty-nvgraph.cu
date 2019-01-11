#include <stdio.h>
#include <nvgraph.h>
#include "common.h"
#include "utils.h"
#define ceiling(a,b) ( (a+b-1)/b )

const char* version_name = "nvGraph";
int NUM, id;
int N, M;
int T = 256;
int W = 1;

typedef struct {
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphTraversalParameter_t traversal_param;
} nvGraph_t;

nvGraph_t* create_nvgraph(const dist_graph_t *graph, traverse_type_t traverse_type);

int *vis, *dis;
int *nowq, *nxtq;
int *v_pos, *e_dst;
int *pre, *in, *deg, *sp;
float *fp;
float *e_weight, *ans, *nxtans;
float *buf;

__global__ void preprocess_kernel( int *v_pos, int *e_dst, int *in, int N )
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    if( x < N ){
        int begin = v_pos[x];
        int end = v_pos[x+1];
        for(int e = begin; e < end; ++e) {
            int v = e_dst[e];
            atomicAdd( &in[v], 1 );
        }
    }
}

void preprocess(dist_graph_t *graph, traverse_type_t traverse_type) {
    if(graph->p_num > 1) {
        printf("not implemented. Only support single-process.\n");
        fatal_error(11);
    }
    graph->additional_info = create_nvgraph(graph, traverse_type);
    N = graph->global_v;
    M = graph->global_e;
    cudaMalloc((void**)&v_pos, sizeof(int)*(N+1) );
    cudaMalloc((void**)&e_dst, sizeof(int)*M );
    cudaMalloc((void**)&e_weight, sizeof(float)*M );
    cudaMalloc((void**)&vis, sizeof(int)*N );
    cudaMalloc((void**)&ans, sizeof(float)*N );
    cudaMalloc((void**)&nxtans, sizeof(float)*N );
    cudaMalloc((void**)&nowq, sizeof(int)*N );
    cudaMalloc((void**)&nxtq, sizeof(int)*N );
    // cudaMalloc((void**)&in, sizeof(int)*N );
    // cudaMalloc((void**)&deg, sizeof(int)*N );
    // cudaMalloc((void**)&buf, sizeof(float)*M );
    // cudaMalloc((void**)&pre, sizeof(int)*M );
    cudaMemcpy( v_pos, graph->v_pos, sizeof(int)*(N+1), cudaMemcpyHostToDevice );
    cudaMemcpy( e_dst, graph->e_dst, sizeof(int)*M, cudaMemcpyHostToDevice );
    sp = (int*)malloc( sizeof(int)*N );
    fp = (float*)malloc( sizeof(float)*N );
    int self_cir = 0, mis_weight = 0, multi_edge = 0;
    // for( int i = 0; i < N; i ++ ){
    //     int begin = graph->v_pos[i];
    //     int end = graph->v_pos[i+1];
    // }
    if( traverse_type == SSSP ){
        cudaMemcpy( e_weight, graph->e_weight, sizeof(float)*M, cudaMemcpyHostToDevice );

        // dim3 grid_size (ceiling(N,T));
        // dim3 block_size (T);
        // preprocess_kernel<<<grid_size, block_size>>>(
        //     v_pos, e_dst, in, N
        // );
        // int *a = (int*)malloc( sizeof(int)*N );
        // cudaMemcpy( a, in, sizeof(int)*N, cudaMemcpyDeviceToHost );
        // int sum = 0;
        // for( int i = 0; i < N; i ++ ){
        //     int tmp = sum;
        //     sum += a[i];
        //     a[i] = tmp;
        // }
        // cudaMemcpy( in, a, sizeof(int)*N, cudaMemcpyHostToDevice );
    }
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

__global__ void init_bfs( int *vis, int *nowq, int *nxtq, int N, int s )
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    if( x < N ){
        if( x == s ){
            vis[x] = x;
            nowq[x] = 0;
        }
        else vis[x] = nowq[x] = -1;
        nxtq[x] = -1;
    }
}

__global__ void bfs_count( int *nowq, int *qsize, int N, int t )
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ int cnt[256];
    if( x < N ){
        if( nowq[x] == t ){
            cnt[threadIdx.x] = 1;
        }
        else cnt[threadIdx.x] = 0;
    }
    else cnt[threadIdx.x] = 0;
    __syncthreads();
    int y = threadIdx.x;
    for( int i = 0; i < 8; i ++ ){
        if( ( y & ((1<<i+1)-1) ) == 0 ){
            cnt[y] += cnt[y^(1<<i)];
        }
         __syncthreads();
    }
    if( y == 0 ){
        atomicAdd( qsize, cnt[y] );
    }
}

__global__ void bfs_kernel( int *v_pos, int *e_dst, 
                             int *vis, int *nowq, int *nxtq, int cnt, 
                             int N, int *changed )
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int flag = 0;
    __shared__ int fl;
    if( threadIdx.x == 0 ) fl = 0;
    __syncthreads();
    if( x < N ){
        if( nowq[x] == cnt ){
            int begin = v_pos[x];
            int end = v_pos[x+1];
            for(int e = begin; e < end; ++e) {
                int v = e_dst[e];
                if( vis[v] == -1 ){
                    vis[v] = x;
                    nxtq[v] = cnt + 1;
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

__global__ void bfs_bottom_kernel( int *v_pos, int *e_dst, 
                             int *vis, int *nowq, int *nxtq, int cnt, 
                             int N, int *changed )
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int flag = 0;
    __shared__ int fl;
    if( threadIdx.x == 0 ) fl = 0;
    __syncthreads();
    if( x < N ){
        if( vis[x] == -1 ) {
            int begin = v_pos[x];
            int end = v_pos[x+1];
            for(int e = begin; e < end; ++e) {
                int v = e_dst[e];
                if( nowq[v] == cnt ){
                    vis[x] = v;
                    nxtq[x] = cnt + 1;
                    flag = 1;
                    break;
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
        int *changed, cnt = 0, tmp = 0, tt = 0, *qsize;
        cudaMalloc((void **) &changed, sizeof(int));
        cudaMalloc((void **) &qsize, sizeof(int));
        cudaMemcpy( changed, &tt, sizeof(int), cudaMemcpyHostToDevice );
        dim3 grid_size (ceiling(N,T));
        dim3 block_size (T);
        init_bfs<<<grid_size, block_size>>>( vis, nowq, nxtq, N, s );
        int sum;
        do{
            /*way 1 copy out for count*/
            // cudaMemcpy( sp, nowq, sizeof(int)*N, cudaMemcpyDeviceToHost );
            // sum = 0;
            // for( int i = 0; i < N; i ++ ){
            //     if( sp[i] == cnt )
            //         sum ++;
            // }

            /*way 2 use gpu to count*/
            // tmp = 0;
            // cudaMemcpy( qsize, &tmp, sizeof(int), cudaMemcpyHostToDevice );
            // bfs_count<<<grid_size, block_size>>>(
            //     nowq, qsize, N, cnt
            // );
            // cudaMemcpy( &tmp, qsize, sizeof(int), cudaMemcpyDeviceToHost );
            // sum = tmp;

            //printf("ite %d size %d my size %d\n", cnt, sum, tmp);
            sum = 0;
            if( sum > N/20 )
                bfs_bottom_kernel<<<grid_size, block_size>>>(
                    v_pos, e_dst, vis, nowq, nxtq, cnt, N, changed
                );
            else
                bfs_kernel<<<grid_size, block_size>>>(
                    v_pos, e_dst, vis, nowq, nxtq, cnt, N, changed
                );
            cnt ++;
            cudaMemcpy( &tmp, changed, sizeof(int), cudaMemcpyDeviceToHost );
            //printf("ite %d %d\n", cnt, tmp);
            cudaMemcpy( changed, &tt, sizeof(int), cudaMemcpyHostToDevice );
            int *bfs = nxtq;
            nxtq = nowq;
            nowq = bfs;
        }while(tmp);

        cudaMemcpy(pred, vis, sizeof(int)*N, cudaMemcpyDeviceToHost);
        //printf("iteration %d %d root %d\n", cnt, s, pred[s]);
    }
}

__global__ void init_sssp( int *vis, float *ans, 
                           int *nowq, int *nxtq,
                           int N, int s )
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    if( x < N ){
        if( x == s ){
            vis[x] = x;
            ans[x] = 0.0;
            nowq[x] = 1;
        }
        else{
            vis[x] = -1;
            ans[x] = INFINITY;
            nowq[x] = 0;
        }
        nxtq[x] = 0;
    }
}

__global__ void sssp_init_ans( float *ans, float *nxtans, int N )
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    if( x < N ){
        nxtans[x] = ans[x];
    }
}

__global__ void sssp_kernel( int *v_pos, int *e_dst, float *e_weight,
                             int *vis, float *ans, float *nxtans,
                             int *nowq, int *nxtq,
                             int cnt, int N, int *changed )
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int flag = 0;
	int y = 0;
    __shared__ int fl;
    if( threadIdx.x == 0 ) fl = 0;
    //__syncthreads();
    if( x < N ){
        int begin = v_pos[x];
        int end = v_pos[x+1];
        for(int e = begin; e < end; ++e) {
            int v = e_dst[e];
            if( nowq[v] == cnt )
                if( nxtans[x] > ans[v] + e_weight[e] ){
                    nxtans[x] = ans[v] + e_weight[e];
                    vis[x] = v;
                    nxtq[x] = cnt+1;
                    flag = 1;
                }
        }
    }
    if( flag ){
        fl = 1;
    }
    __syncthreads();
    if( threadIdx.x == 0 && fl ) *changed = 1;
}

__global__ void sssp_reduction( int *vis, float *ans,
                                int *in, int *deg, float *buf, 
                                int *pre, int *nowq, int *nxtq,
                                int N, int cnt )
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    float min;
    int fa;
    min = INFINITY;
    fa = -1;
    if( x < N ){
        for( int i = in[x]; i < in[x]+deg[x]; i ++ ){
            if( buf[i] < min ){
                min = buf[i];
                fa = pre[i];
            }
        }
        if( deg[x] != 0 ){
            ans[x] = min;
            vis[x] = fa;
            nxtq[x] = cnt+1;
        }
        deg[x] = 0;
    }
}

void sssp(dist_graph_t *graph, index_t s, index_t* pred, weight_t* distance){
    if(graph->p_id == 0){
        nvGraph_t* nvgraph = (nvGraph_t*)graph->additional_info;
        int *changed, cnt = 0, tmp = 0, tt = 0;
        cudaMalloc((void **) &changed, sizeof(int));
        cudaMemcpy( changed, &tt, sizeof(int), cudaMemcpyHostToDevice );
        dim3 grid_size (ceiling(N,T));
        dim3 block_size (T);
        // int begin = graph->v_pos[s];
        // int end = graph->v_pos[s+1];
        // printf("%d: ", s);
        // for( int e = begin; e < end; e ++ ){
        //     printf("%d %f\t", graph->e_dst[e], graph->e_weight[e]);
        // }
        // puts("");
        init_sssp<<<grid_size, block_size>>>( vis, ans, nowq, nxtq, N, s );
        int num;
        do{
            cnt ++;
            sssp_init_ans<<<grid_size, block_size>>>( ans, nxtans, N );
            // printf("it %d", cnt);
            cudaMemcpy( sp, nowq, sizeof(int)*N, cudaMemcpyDeviceToHost );
            // num = 0;
            // for( int i = 0; i < N; i ++ ){
            //     if( sp[i] == cnt ){
            //         //printf("%d ", i);
            //         num ++;
            //     }
            // }
            // printf(" num %d\n", num);
            
            // printf("ans\n");
            // num = 0;
            // cudaMemcpy( fp, ans, sizeof(float)*N, cudaMemcpyDeviceToHost );
            // for( int i = 0; i < N; i ++ ){
                // if( fp[i] != INFINITY ){
                    // printf("%d : %f\n", i, fp[i]);
                    // num ++;
                // }
            // }
            // printf("num %d\n", num);
            sssp_init_ans<<<grid_size, block_size>>>( ans, nxtans, N );
            sssp_kernel<<<grid_size, block_size>>>(
                v_pos, e_dst, e_weight, vis, ans, nxtans,
                nowq, nxtq, cnt, N, changed
            );
            //printf("nxtans\n");
            // num = 0;
            // cudaMemcpy( fp, nxtans, sizeof(float)*N, cudaMemcpyDeviceToHost );
            // for( int i = 0; i < N; i ++ ){
            //     if( fp[i] != INFINITY ){
            //         printf("%d : %f\n", i, fp[i]);
            //         num ++;
            //     }
            //     printf("%d %6d %.6f\n", cnt, i, fp[i]);
            // }
            // printf("num %d\n", num);

            // int debug[10];
            // cudaMemcpy( debug, lock, sizeof(int)*2, cudaMemcpyDeviceToHost );
            // printf("d0 %d d1 %d\n", debug[0], debug[1]);

            cudaMemcpy( &tmp, changed, sizeof(int), cudaMemcpyDeviceToHost );
            //printf("ite %d %d\n", cnt, tmp);
            cudaMemcpy( changed, &tt, sizeof(int), cudaMemcpyHostToDevice );
            int *tq = nowq;
            nowq = nxtq;
            nxtq = tq;
            float *ftp = ans;
            ans = nxtans;
            nxtans = ftp;
        }while(tmp);

        cudaMemcpy(pred, vis, sizeof(int)*N, cudaMemcpyDeviceToHost);
        cudaMemcpy(distance, ans, sizeof(float)*N, cudaMemcpyDeviceToHost);
        //printf("iteration %d\n", cnt);
    }
}