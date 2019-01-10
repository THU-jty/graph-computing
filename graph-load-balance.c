#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "common.h"
#include "utils.h"
#define mem0(a,b) for(int kk=0;kk<b;kk++){a[kk]=0;}
#define mem1(a,b) for(int kk=0;kk<b;kk++){a[kk]=-1;}

//#define __DEBUG
#ifdef __DEBUG
#define DEBUG(info,...)    printf(info, ##__VA_ARGS__)
#else
#define DEBUG(info,...)
#endif

const char* version_name = "A reference version of edge-based load balancing";
const int T = 70;
int loc[T], NUM, id, N, M, *rev, *bt_recv, *mark;

/* C++ std::lower_bound */
int lower_bound(const index_t* a, int left, int right, index_t value) {
    int size = right - left;
    while (size > 0) {
        int step = size / 2; 
        if (a[left + step] < value) {
            left += step + 1; 
            size -= step + 1; 
        } else {
            size = step;
        }
    }
    return left;
}

void swap( int *a, int *b ){
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

/* 
 * old partition: number of vertices on each process is balanced
 * new partition: number of edges on each process is balanced
 */
void preprocess(dist_graph_t *graph, traverse_type_t traverse_type) {
    int p_id = graph->p_id, p_num = graph->p_num;
    int global_e = graph->global_e, local_e = graph->local_e;
    int local_v = graph->local_v, offset_v = graph->offset_v;
    int q = global_e / p_num;
    int r = global_e % p_num;
    int e_part = p_id * q + ((p_id < r) ? p_id : r);
    index_t* v_pos = graph->v_pos;
    index_t* e_dst = graph->e_dst;
    weight_t* e_weight = graph->e_weight;
    int new_local_v, new_offset_v, new_local_e;
    index_t* new_v_pos;
    index_t* new_e_dst;
    weight_t* new_e_weight = NULL;
    index_t* e_partitions = (index_t*) malloc(sizeof(index_t) * (p_num * 9));
    index_t* v_sendcounts = e_partitions + p_num * 1;
    index_t* v_recvcounts = e_partitions + p_num * 2;
    index_t* v_sdispls = e_partitions + p_num * 3;
    index_t* v_rdispls = e_partitions + p_num * 4;
    index_t* e_sendcounts = e_partitions + p_num * 5;
    index_t* e_recvcounts = e_partitions + p_num * 6;
    index_t* e_sdispls = e_partitions + p_num * 7;
    index_t* e_rdispls = e_partitions + p_num * 8;

    /* ask other processes: which part of data do you want? */
    MPI_Allgather(&e_part, 1, MPI_INT, e_partitions, 1, MPI_INT, MPI_COMM_WORLD);

    /* determine: how should I send my data to other processes? */
    int partition_point = 0;
    for(int p = 0; p < p_num; ++ p) { 
        partition_point = lower_bound(v_pos, partition_point, local_v, e_partitions[p]);
        v_sdispls[p] = partition_point;
        e_sdispls[p] = v_pos[partition_point] - v_pos[0];
    }
    for(int p = 0; p < p_num - 1; ++ p) {
        v_sendcounts[p] = v_sdispls[p+1] - v_sdispls[p];
        e_sendcounts[p] = e_sdispls[p+1] - e_sdispls[p];
    }
    v_sendcounts[p_num - 1] = local_v - v_sdispls[p_num - 1];
    e_sendcounts[p_num - 1] = local_e - e_sdispls[p_num - 1];

    /* ask other processes: how should I receive data from you? */
    MPI_Alltoall(v_sendcounts, 1, MPI_INT,
                 v_recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoall(e_sendcounts, 1, MPI_INT,
                 e_recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    int v_sum = 0, e_sum = 0;
    for(int p = 0; p < p_num; ++p) {
        v_rdispls[p] = v_sum;
        e_rdispls[p] = e_sum;
        v_sum += v_recvcounts[p];
        e_sum += e_recvcounts[p];
    }
    new_local_v = v_sum;
    new_local_e = e_sum;

    /* shuffle graph data */
    new_v_pos = (index_t*)malloc(sizeof(index_t) * (new_local_v + 1));
    MPI_Alltoallv(v_pos, v_sendcounts, v_sdispls, MPI_INT, \
              new_v_pos, v_recvcounts, v_rdispls, MPI_INT, MPI_COMM_WORLD);
    free(v_pos);
    new_e_dst = (index_t*)malloc(sizeof(index_t) * new_local_e);
    MPI_Alltoallv(e_dst, e_sendcounts, e_sdispls, MPI_INT, \
              new_e_dst, e_recvcounts, e_rdispls, MPI_INT, MPI_COMM_WORLD);
    free(e_dst);
    if(traverse_type == SSSP) {
        new_e_weight = (weight_t*)malloc(sizeof(weight_t) * new_local_e);
        MPI_Alltoallv(e_weight, e_sendcounts, e_sdispls, MPI_FLOAT, \
                  new_e_weight, e_recvcounts, e_rdispls, MPI_FLOAT, MPI_COMM_WORLD);
        free(e_weight);
    }
    free(e_partitions);

    /* miscellaneous information */
    new_offset_v = 0;
    MPI_Exscan(&new_local_v, &new_offset_v, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    new_v_pos[new_local_v] = new_v_pos[0] + new_local_e;

    graph->local_v = new_local_v;
    graph->offset_v = new_offset_v;
    graph->local_e = new_local_e;
    graph->v_pos = new_v_pos;
    graph->e_dst = new_e_dst;
    graph->e_weight = new_e_weight;

    NUM = graph->p_num;
    id = graph->p_id;
    void **p = malloc( sizeof(void*)*10 );
    p[0] = malloc( (graph->global_v*2)*sizeof(int) );
    p[1] = malloc( (graph->local_v*2*NUM)*sizeof(int) );
    p[2] = malloc( (graph->global_v)*sizeof(int) );
    p[3] = malloc( (graph->global_v)*sizeof(int) );
    p[4] = malloc( graph->p_num*3*sizeof(int) );
    p[5] = malloc( graph->global_v*sizeof(float) );
    p[6] = malloc( graph->local_v*NUM*sizeof(float) );
    p[7] = malloc( graph->global_v*sizeof(float) );
    p[8] = malloc( graph->global_v*sizeof(int) );
    mark = (int*)malloc( graph->global_v*sizeof(int) );
    bt_recv = (int*)malloc( (graph->global_v*2)*sizeof(int) );
    rev = (int*)malloc( (graph->global_v)*sizeof(int) );
    int tmp[1000];
    for( int i = 0; i < graph->p_num; i ++ ){
        tmp[3*i] = new_local_v;
        tmp[3*i+1] = new_offset_v;
        tmp[3*i+2] = new_local_e;
    }
    MPI_Alltoall(tmp,3,MPI_INT,
                 p[4],3,MPI_INT,MPI_COMM_WORLD);
    //DEBUG("pre %d %p %p %p %p\n", id, p[0], p[1], p[2], p[3]);

    int *pint = (int*)p[4];
    for( int i = 0; i < NUM; i ++ ) loc[i] = pint[3*i+1];
    loc[NUM] = graph->global_v;    
    N = graph->global_v;
    M = graph->global_e;
    graph->additional_info = p;
    
    for( int i = 0; i < NUM; i ++ ){
        for( int j = loc[i]; j < loc[i+1]; j ++ )
        rev[j] = i;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printf("\e[1;32m%d before: local_v = %d, offset_v = %d, local_e = %d\e[0m\n", \
                        p_id, local_v, offset_v, local_e);
    printf("\e[1;34m%d after : local_v = %d, offset_v = %d, local_e = %d\e[0m\n", \
                        p_id, new_local_v, new_offset_v, new_local_e);
    MPI_Barrier(MPI_COMM_WORLD);
    
}

void destroy_additional_info(void *additional_info) {
    free(bt_recv);
}

inline int belong( int v )
{
    //return id;
    return rev[v];
    // if( loc[0] <= v && v < loc[1] ) return 0;
    // if( loc[1] <= v && v < loc[2] ) return 1;
    // if( loc[2] <= v && v < loc[3] ) return 2;
    // if( loc[3] <= v && v < loc[4] ) return 3;
    // if( loc[4] <= v && v < loc[5] ) return 4;
    // if( loc[5] <= v && v < loc[6] ) return 5;
    // if( loc[6] <= v && v < loc[7] ) return 6;
    // if( loc[7] <= v && v < loc[8] ) return 7;
    // if( loc[8] <= v && v < loc[9] ) return 8;
    // if( loc[9] <= v && v < loc[10] ) return 9;
    // for( int i = NUM-1; i >= 0; i -- )
    //     if( v >= loc[i] ) return i;
    // int l = 0, r = NUM-1, mid, ans = -1;
    // while( l <= r ){
    //     mid = (l+r)/2;
    //     if( loc[mid] <= v ){
    //         l = mid+1;
    //         ans = mid;
    //     }
    //     else r = mid-1;
    // }
    // return ans;
    // for( int i = 0; i < NUM; i ++ ){
    //     if( loc[i] <= v && v < loc[i+1] ) return i;
    // }
}

void bfs(dist_graph_t *graph, index_t s, index_t* pred){
    const index_t* v_pos = graph->v_pos;
    const index_t* e_dst = graph->e_dst;
    const int offset_v = graph->offset_v, local_v = graph->local_v;
    const int global_v = graph->global_v;
    int offset_e = graph->v_pos[0];
    int *send_buf[T], *recv_buf[T], *p[10], *vis, sdis[T], rdis[T], recv_flag[T];
    void **q = graph->additional_info;
    for( int i = 0; i < 4; i ++ ){
        p[i] = (*q);
        q += 1;
    }
    //DEBUG("bfs %d %p %p %p %p\n", id, p[0], p[1], p[2], p[3]);
    //MPI_Barrier(MPI_COMM_WORLD);
    //DEBUG("%d: ", id);for( int i = 0; i <= NUM; i ++ ) DEBUG("%d ", loc[i]); DEBUG("\n");
    //MPI_Barrier(MPI_COMM_WORLD);
    for( int i = 0; i < NUM; i ++ ){
        send_buf[i] = p[0]+loc[i]*2;
        recv_buf[i] = p[1]+i*graph->local_v*2;
        sdis[i] = loc[i]*2;
        rdis[i] = i*graph->local_v*2;
		if( id != -1 ){
			DEBUG("%d %d send_buf %p recv_buf %p\n", id, i, send_buf[i], recv_buf[i]);
		}
    }
    if( NUM != 1 ){
        vis = p[2];
        mem1( vis, N );
    }
    else{
        vis = pred;
        for(int i = 0; i < local_v; ++i) {
           pred[i] = UNREACHABLE;
        }
    }
    int send_cnt[T], recv_cnt[T];
    int cnt[T];
    mem0( cnt, 0 );

    index_t* queue_f = p[3];
    index_t* queue_g = send_buf[id];
    int size_f = 0;
    int size_g = 0;
    int i, j, k;

    if( belong( s ) == id ){
        pred[s-offset_v] = s;
        queue_f[size_f++] = s;
		vis[s] = s;
    }
    int ite = 0;
    do {
        ite ++;
        //printf("%d it %d qsize %d\n", id, ite, size_f);
        mem0( send_cnt, NUM );
        index_t* tmp;
        int size_g = 0;
        int send_flag = 0;

        send_flag = size_f;
        if( NUM != 1 )
        MPI_Allgather( &send_flag, 1, MPI_INT,
        recv_flag, 1, MPI_INT, MPI_COMM_WORLD );
        int sum = 0;
        for( int i = 0; i < NUM; i ++ ){
            sum += recv_flag[i];
        }
        if( NUM == 1 ) sum = size_f;
        if( sum == 0 ) break;
        if( sum > N/10 ){
            //printf("%d bottom\n", id);
            if( NUM != 1 ){
                MPI_Allgatherv( queue_f, size_f, MPI_INT,
                bt_recv, recv_flag, sdis, MPI_INT, MPI_COMM_WORLD );
            }
            //printf("%d gather\n", id);
            for( int i = 0; i < NUM; i ++ ){
                if( i == id ) continue;
                for( int j = 0; j < recv_flag[i]; j ++ ){
                    if( bt_recv[j+sdis[i]] < 0 || bt_recv[j+sdis[i]] >= N ) printf("%d\n", bt_recv[j+sdis[i]]);
                    vis[bt_recv[j+sdis[i]]] = 0;
                }
            }
            //printf("%d update\n", id);
            for(int i = offset_v; i < offset_v+local_v; ++i){
                int u = i;
                if( vis[u] != -1 ) continue;
                int begin = v_pos[u-offset_v] - v_pos[0];
                int end = v_pos[u+1-offset_v] - v_pos[0];
                for(int e = begin; e < end; ++e) {
                    int v = e_dst[e];
                    if( vis[v] != -1 ){
                        //vis[u] = v;
                        queue_g[ size_g++ ] = u;
                        mark[u] = v;
                        break;
                    }
                }
            }
            for( int i = 0; i < size_g; i ++ ){
                vis[ queue_g[i] ] = mark[ queue_g[i] ];
            }
            tmp = queue_g;
            queue_g = queue_f;
            queue_f = tmp;
            size_f = size_g;
        }
        else{
            send_flag = 0;
            for(int i = 0; i < size_f; ++i){
                int u = queue_f[i];
                int begin = v_pos[u-offset_v] - v_pos[0];
                int end = v_pos[u+1-offset_v] - v_pos[0];
                for(int e = begin; e < end; ++e) {
                    int v = e_dst[e];
                    if( vis[v] == -1 ){
                        vis[v] = u;
                        send_flag = 1;
                        int bg = belong(v);
                        if( bg != id ){
                            send_buf[bg][ send_cnt[bg]*2 ] = v;
                            send_buf[bg][ send_cnt[bg]*2+1 ] = u;
                            send_cnt[bg] ++;
                        }
                        else{
                            queue_g[ send_cnt[bg] ] = v;
                            send_cnt[bg] ++;
                        }
                    }
                }
            }
            DEBUG("%d it %d exec over\n", id, ite);
            
            // MPI_Barrier(MPI_COMM_WORLD);
            // DEBUG("send id %d it %d: ", id, ite);
            // for( int i = 0; i < NUM; i ++ ){
            //      DEBUG("%d ", send_cnt[i]);
            // }
            // DEBUG("\n");
            // MPI_Barrier(MPI_COMM_WORLD);
            
            //notify all nodes over or not
            // if( NUM != 1 )
            // MPI_Allgather( &send_flag, 1, MPI_INT,
            // recv_flag, 1, MPI_INT, MPI_COMM_WORLD );
            // for( i = 0; i < NUM; i ++ ){
            //     if( recv_flag[i] != 0 ) break;
            // }
            // if( i == NUM && NUM != 1 ){
            //     DEBUG("%d it %d end!!!\n", id, ite);
            //     break;
            // }
            // if( NUM == 1 && size_f == 0 ){
            //     DEBUG("%d it %d end!!!\n", id, ite);
            //     break;
            // }

            int send_self = 0;
            swap( &send_self, &send_cnt[id] );
            if( NUM != 1 )
            MPI_Alltoall(send_cnt,1,MPI_INT,
                recv_cnt,1,MPI_INT,MPI_COMM_WORLD);
            DEBUG("%d it %d cnt over\n", id, ite);
            // MPI_Barrier(MPI_COMM_WORLD);
            // DEBUG("recv id %d it %d: ", id, ite);
            // for( int i = 0; i < NUM; i ++ ){
            //     DEBUG("%d ", recv_cnt[i]);
            // }
            // DEBUG("\n");
            // MPI_Barrier(MPI_COMM_WORLD);
            
            for( int i = 0; i < NUM; i ++ ){
                send_cnt[i] *= 2;
                recv_cnt[i] *= 2;
            }
            if( NUM != 1 )
            MPI_Alltoallv(send_buf[0],send_cnt,sdis,MPI_INT,
                        recv_buf[0],recv_cnt,rdis,MPI_INT,
                        MPI_COMM_WORLD);
            //printf("ppp %d\n", id);
            // for( int i = 0; i < NUM; i ++ ){
            // 	for( int j = 0; j < 10; j ++ ){
            // 		printf("id %d : %d %d %d\n", id, i, j, recv_buf[i][j]);
            // 	}
            // }
            DEBUG("%d it %d data over\n", id, ite);
            size_f = 0;
            for( int i = 0; i < NUM; i ++ ){
                if( i != id ){
                    for( int j = 0; j < recv_cnt[i]/2; j ++ ){
                        int v = recv_buf[i][j*2];
                        if( vis[v] == -1 ){
                            queue_g[send_self++] = v;
                            //queue_f[size_f++] = v;
                            vis[v] = recv_buf[i][j*2+1];
                            //printf("%d : %d %d\n", id, v, recv_buf[i][j*2+1]);
                            vis[ recv_buf[i][j*2+1] ] = 1;
                        }
                    }
                }
                // else{
                //     for( int j = 0; j < send_self; j ++ ){
                //         int v = send_buf[i][j];
                //         queue_f[size_f++] = v;
                //     }
                // }
            }
            size_f = send_self;
            tmp = queue_g;
            queue_g = queue_f;
            queue_f = tmp;
        }
    } while(1);   
    DEBUG("%d it %d sta %d\n", id, ite, s);
    if( id == 0 ){
        //printf("iterations %d st %d\n", ite, s);
    }
    for( int i = 0; i < local_v; i ++ ){
        pred[i] = vis[i+offset_v];
    }
    // if( id == 0 && s == 179840 ){
    //     for( int i = 0; i < local_v; i ++ ){
    //         //printf("%d %d\n", i+offset_v, vis[i+offset_v]);
    //     }
    // }
}


void sssp(dist_graph_t *graph, index_t s, index_t* pred, weight_t* distance){
    const index_t* v_pos = graph->v_pos;
    const index_t* e_dst = graph->e_dst;
    const int offset_v = graph->offset_v, local_v = graph->local_v;
    const int global_v = graph->global_v;
    const weight_t* e_weight = graph->e_weight;
    int offset_e = graph->v_pos[0];
    int *send_buf[T], *recv_buf[T], *p[10], *vis, sdis[T], \
    rdis[T], recv_flag[T], sf[T], rf[T], *stp;
    float *send_v[T], *recv_v[T], *dis;
    void **q = graph->additional_info;
    for( int i = 0; i < 9; i ++ ){
        p[i] = (*q);
        q += 1;
    }
    //DEBUG("bfs %d %p %p %p %p\n", id, p[0], p[1], p[2], p[3]);
    //MPI_Barrier(MPI_COMM_WORLD);
    //DEBUG("%d: ", id);for( int i = 0; i <= NUM; i ++ ) DEBUG("%d ", loc[i]); DEBUG("\n");
    //MPI_Barrier(MPI_COMM_WORLD);
    for( int i = 0; i < NUM; i ++ ){
        send_buf[i] = p[0]+loc[i]*2;
        recv_buf[i] = p[1]+i*graph->local_v*2;
        float* tmp = (float*)p[5];
        send_v[i] = tmp+loc[i];
        tmp = (float*)p[6];
        //if( i == 1 )printf("tmp %p tmp+ %p %d\n", tmp, tmp+loc[i], loc[i]);
        recv_v[i] = tmp+i*graph->local_v;
        sdis[i] = loc[i]*2;
        rdis[i] = i*graph->local_v*2;
        sf[i] = loc[i];
        rf[i] = i*graph->local_v;
		if( id != -1 ){
			DEBUG("%d %d send_buf %p recv_buf %p\n", id, i, send_buf[i], recv_buf[i]);
		}
    }
    stp = p[8];
    mem0( stp, N );
    if( NUM != 1 ){
        vis = p[2];
        dis = (float*)p[7];
        mem1( vis, N );
        for( int i = 0; i < N; i ++ ){
            dis[i] = INFINITY;
        }
    }
    else{
        vis = pred;
        dis = distance;
        for(int i = 0; i < local_v; ++i) {
           distance[i] = INFINITY;
           pred[i] = UNREACHABLE;
        }
    }
    int send_cnt[T], recv_cnt[T];

    index_t* queue_f = p[3];
    index_t* queue_g = send_buf[id];
    int size_f = 0;
    int size_g = 0;
    int i, j, k;

    vis[s] = s;
    dis[s] = 0.0;
    if( belong( s ) == id ){
        pred[s-offset_v] = s;
        queue_f[size_f++] = s;
    }
    int ite = 0;
    int *fl = malloc( sizeof(int)*N );
    do {
        //mem0(fl, N);
        ite ++;
        DEBUG("%d it %d qsize %d\n", id, ite, size_f);      
        mem0( send_cnt, NUM );
        index_t* tmp;
        int size_g = 0;
        int send_flag = 0;
        int send_self = 0;
        for(int i = 0; i < size_f; ++i){
            int u = queue_f[i];
            int begin = v_pos[u-offset_v] - v_pos[0];
            int end = v_pos[u+1-offset_v] - v_pos[0];
            for(int e = begin; e < end; ++e) {
                int v = e_dst[e];
                if( dis[v] > dis[u] + e_weight[e] ){                  
                    vis[v] = u;
                    dis[v] = dis[u] + e_weight[e];
                    send_flag = 1;
                    int bg = belong(v);
                    if( bg != id ){
                        if( stp[v] != ite ){
                            send_buf[bg][ send_cnt[bg]*2 ] = v;
                            send_cnt[bg] ++;
                            stp[v] = ite;
                        }
                    }
                    else{
                        if( stp[v] != ite ){
                            queue_g[ send_self ] = v;
                            send_self ++;
                            stp[v] = ite;
                        }
                    }
                }
            }
        }
        for( int i = 0; i < NUM; i ++ ){
            if( i != id ){
                for( int j = 0; j < send_cnt[i]; j ++ ){
                    send_buf[ i ][ j*2+1 ] = vis[send_buf[ i ][ j*2 ]];
                    send_v[i][j] = dis[send_buf[ i ][ j*2 ]];
                }
            }
        }
        //DEBUG("%d it %d exec over\n", id, ite);
	    
        // MPI_Barrier(MPI_COMM_WORLD);
        // DEBUG("send id %d it %d: ", id, ite);
        // for( int i = 0; i < NUM; i ++ ){
        //      DEBUG("%d ", send_cnt[i]);
        // }
		// DEBUG("\n");
        // MPI_Barrier(MPI_COMM_WORLD);
        
        //notify all nodes over or not
        if( NUM != 1 )
        MPI_Allgather( &send_flag, 1, MPI_INT,
        recv_flag, 1, MPI_INT, MPI_COMM_WORLD );
        for( i = 0; i < NUM; i ++ ){
            if( recv_flag[i] != 0 ) break;
        }
        if( i == NUM && NUM != 1 ){
            DEBUG("%d it %d end!!!\n", id, ite);
            break;
        }
        if( NUM == 1 && size_f == 0 ){
            DEBUG("%d it %d end!!!\n", id, ite);
            break;
        }

        if( NUM != 1 )
        MPI_Alltoall(send_cnt,1,MPI_INT,
			recv_cnt,1,MPI_INT,MPI_COMM_WORLD);
		//DEBUG("%d it %d cnt over\n", id, ite);
        // MPI_Barrier(MPI_COMM_WORLD);
		// DEBUG("recv id %d it %d: ", id, ite);
		// for( int i = 0; i < NUM; i ++ ){
        //     DEBUG("%d ", recv_cnt[i]);
        // }
        // DEBUG("\n");
        // MPI_Barrier(MPI_COMM_WORLD);
		
		for( int i = 0; i < NUM; i ++ ){
            if( send_cnt[i] > loc[i+1]-loc[i] ) printf("%d %d error %d %d\n", id, ite, i, send_cnt[i]);
            send_cnt[i] *= 2;
            recv_cnt[i] *= 2;
        }
        if( NUM != 1 )
        MPI_Alltoallv(send_buf[0],send_cnt,sdis,MPI_INT,
                    recv_buf[0],recv_cnt,rdis,MPI_INT,
                    MPI_COMM_WORLD);
        for( int i = 0; i < NUM; i ++ ){
            send_cnt[i] /= 2;
            recv_cnt[i] /= 2;
            if( recv_cnt[i] > local_v ) printf("%d %d error %d %d\n", id, ite, i, recv_cnt[i]);
        }
        if( NUM != 1 )
        MPI_Alltoallv(send_v[0],send_cnt,sf,MPI_FLOAT,
                    recv_v[0],recv_cnt,rf,MPI_FLOAT,
                    MPI_COMM_WORLD);
        //printf("ppp %d\n", id);
		// for( int i = 0; i < NUM; i ++ ){
		// 	for( int j = 0; j < 10; j ++ ){
		// 		printf("id %d : %d %d %d\n", id, i, j, recv_buf[i][j]);
		// 	}
		// }
        //DEBUG("%d it %d data over\n", id, ite);
        size_f = 0;
        for( int i = 0; i < NUM; i ++ ){
            if( i != id ){
                for( int j = 0; j < recv_cnt[i]; j ++ ){
                    int v = recv_buf[i][j*2];
                    //if( belong(v) != id ){ printf("error\n"); exit(0); }
                    if( dis[v] > recv_v[i][j] ){
                        int u = recv_buf[i][j*2+1];
                        dis[v] = recv_v[i][j];
                        vis[v] = recv_buf[i][j*2+1];
                        if( stp[v] != ite ){
                            queue_g[send_self++] = v;
                            stp[v] = ite;
                            //printf("%d : %d %d\n", id, v, recv_buf[i][j*2+1]);
                        }
                    } 
                }
            }
        }
        size_f = send_self;
        tmp = queue_g;
        queue_g = queue_f;
        queue_f = tmp;
    } while(1);   
    //DEBUG("%d it %d\n", id, ite);
    if( id == 0 ){
        //printf("iterations %d\n", ite);
    }
    //sleep(id*5);
    if( NUM != 1 )
    for( int i = 0; i < local_v; i ++ ){
        pred[i] = vis[i+offset_v];
        distance[i] = dis[i+offset_v];
    }
}
/*
记录一下，一天就调了两个bug(12.29)
1 memset vis的时候memset成0而不是-1,导致bfs tree成环了
2 给recv float buffer开空间的时候直接分配了global_v而不是local_v*NUM...
启发：尽早进行输出调试，两个bug都是依靠输出调试解决的
第一个输出了所有的前驱发现怎么这么多0啊啊啊啊
第二个依靠每次迭代的集合，输出了集合内容，找到了缓冲区的overlap
*/