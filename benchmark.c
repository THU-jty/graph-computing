#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <stdbool.h>
#include "common.h"
#include "utils.h"

extern const char* version_name;

int parse_args(traverse_type_t* traverse_type, int* reps, int p_id, int argc, char **argv);
const char* result_error_msg(int err);
int my_abort(int line, int code);

#define MY_ABORT(ret) my_abort(__LINE__, ret)
#define ABORT_IF_ERROR(ret) CHECK_ERROR(ret, MY_ABORT(ret))
#define ABORT_IF_NULL(ret) CHECK_NULL(ret, MY_ABORT(MPI_ERR_NO_MEM))
#define INDENT "    "

int main(int argc, char **argv) {
    int reps, done, trivial, thread_scheme, p_id, ret, s;
    int64_t te_total, te;
    traverse_type_t type;
    dist_graph_t graph;
    double start, end, compute_time = 0, pre_time;
    index_t *pred;
    weight_t *distance = NULL;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_scheme);
    MPI_Comm_rank(MPI_COMM_WORLD, &graph.p_id);
    MPI_Comm_size(MPI_COMM_WORLD, &graph.p_num);
    ret = parse_args(&type, &reps, graph.p_id, argc, argv);
    ABORT_IF_ERROR(ret)

    ret = read_graph_default(&graph, argv[3], type);
    ABORT_IF_ERROR(ret)

    if(graph.p_id == 0) {
        printf("Benchmarking %s.\n", version_name);
        printf("Running %s on %s.\n", (type == BFS) ? "BFS" : "SSSP", argv[3]);
        printf(INDENT"%d vertices, %d edges, %d run(s)\n", graph.global_v, graph.global_e, reps);
        printf(INDENT"Preprocessing.\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    preprocess(&graph, type);
    MPI_Barrier(MPI_COMM_WORLD); 
    pre_time = MPI_Wtime() - start;

    pred = (index_t*) malloc(sizeof(index_t) * graph.local_v);
    ABORT_IF_NULL(pred)
    if(type == SSSP) {
        distance = (weight_t*) malloc(sizeof(weight_t) * graph.local_v);
        ABORT_IF_NULL(distance)
    }

    srand((uint32_t)123456789); /* fixed seed, reproducible */
    s = rand() % graph.global_v;
    MPI_Bcast(&s, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* warm up */
    if(graph.p_id == 0) {
        printf(INDENT"Warming up.\n");
    }
    if(type == BFS) {
        bfs(&graph, s, pred);
    } else {
        sssp(&graph, s, pred, distance);
    }

    if(graph.p_id == 0) {
        printf(INDENT"Testing.\n");
    }
    te_total = 0;
    done = 0;
    trivial = 0;
    while(done < reps && trivial < reps) {
        s = rand() % graph.global_v;
        MPI_Bcast(&s, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD); 
        start = MPI_Wtime();
        if(type == BFS) {
            bfs(&graph, s, pred);
        } else {
            sssp(&graph, s, pred, distance);
        }
        MPI_Barrier(MPI_COMM_WORLD); 
        end = MPI_Wtime();
        te = get_te(&graph, pred);
        if(te > (graph.global_e >> 1)) {
            compute_time += end - start;
            te_total += te;
            ++done;
        } else {
            ++trivial;
        }
    }

    if(graph.p_id == 0) {
        if(trivial > 0) {
            printf(INDENT"%d run(s) are trivial.\n", trivial);
            if(done < reps) {
                printf(INDENT"Too many trivial runs.\n");
            }
        }
        printf(INDENT"Checking.\n");
    }
    ret = check_answer(&graph, argv[3], s, type, pred, distance);
    if(graph.p_id == 0) {
        if(ret == 0) {
            printf("\e[1;32m"INDENT"Result validated.\e[0m\n");
        } else {
            printf("\e[1;31m"INDENT"Result NOT validated.\e[0m\n");
            printf("\e[1;31m"INDENT"%s\e[0m\n", result_error_msg(ret));
            MY_ABORT(ret);
        }
    }
    destroy_dist_graph(&graph);

    if(graph.p_id == 0) {
        double mteps = 5e-7 * te_total / compute_time; /* symmetric: divided by 2 */
        double compute_time_per_run = compute_time / reps;
        printf(INDENT INDENT"preprocess time = %lf s, compute time = %lf s per run\n", \
                    pre_time, compute_time_per_run);
        printf("\e[1;34m"INDENT INDENT"Performance: %lf MTEPS\e[0m\n", mteps);
    }

    if(type == SSSP) {
        free(distance);
    }
    free(pred);
    MPI_Finalize();
    return 0;
}

void print_help(const char *argv0, int p_id) {
    if(p_id == 0) {
        printf("\e[1;31mUSAGE: %s <traverse-type> <repetitions> <input-file>\n  where <traverse-type> is \"bfs\" or \"sssp\"\e[0m\n", argv0);
    }
}

int parse_args(traverse_type_t* traverse_type, int* reps, int p_id, int argc, char **argv) {
    int r;
    if(argc < 4) {
        print_help(argv[0], p_id);
        return MPI_ERR_ARG;
    }
    r = atoi(argv[2]);
    if(strcmp(argv[1], "bfs") == 0 || strcmp(argv[1], "BFS") == 0) {
        *traverse_type = BFS;
        *reps = r;
        return MPI_SUCCESS;
    } else if(strcmp(argv[1], "sssp") == 0 || strcmp(argv[1], "SSSP") == 0) {
        *traverse_type = SSSP;
        *reps = r;
        return MPI_SUCCESS;
    } else {
        print_help(argv[0], p_id);
        return MPI_ERR_ARG;
    }
}

int my_abort(int line, int code) {
    printf("\e[1;33merror at line %d, error code = %d\e[0m\n", line, code);
    return fatal_error(code);
}

const char* result_error_msg(int err) {
    static const char *error_msg[] = {
        "Exceptions occurred while checking result.",
        "Non-disjoint vertex partition.",
        "Predecessor index out of bound.",
        "The predecessor of a vertex is unreachable.",
        "The predecessor of the root must be itself.",
        "The traversal tree has loop.",
        "A reachable vertex is marked as \"unreachable\".",
        "The path to a vertex is not the shortest.",
        "For some vertex, its distance is not equal to that of its predecessor plus the edge weight."
    };
    if(err >= 1 && err <= 8) {
        return error_msg[err];
    } else {
        return error_msg[0];
    }
}