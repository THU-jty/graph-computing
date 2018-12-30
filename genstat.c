#include<stdbool.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

char *find_name(char *path) {
    char *p = path + strlen(path);
    while(*p != '/') {
        if(*p == '.') {
            *p = '\0';
        }
        --p;
    }
    return p + 1;
}

double square(double x) {
    return x * x;
}

void write_stats(FILE *file, const char *name, int nv, const int *v_pos, bool head) {
    double stddev = 0.0, avg = (double) v_pos[nv] / (double) nv;
    int i, m = v_pos[1] - v_pos[0];
    for(i = 0; i < nv; ++i) {
        int d = v_pos[i+1] - v_pos[i];
        stddev += square(d - avg);
        if(m < d) {
            m = d;
        }
    }
    stddev = sqrt(stddev / nv);
    if(head) {
        fprintf(file, "%25s,%15s,%15s,%15s,%15s,%20s\n", \
                      "name", "vertices", "edges",  \
                      "maximum degree", "average degree", "standard deviation");
    }
    fprintf(file, "%25s,%15d,%15d,%15d,%15lf,%20lf\n", \
                   name, nv, v_pos[nv], m, avg,stddev);
}

int main(int argc, char **argv) {
    if(argc > 2) {
        int nv;
        int *v_pos;
        char *str;
        FILE *file;
        bool head = (argc > 3) && (strcmp(argv[3], "-h") == 0);

        file = fopen(argv[1], "rb");
        if(file == NULL) {
            printf("Failed to open %s\n", argv[1]);
        }
        fread(&nv, sizeof(int), 1, file);
        v_pos = (int*)malloc(sizeof(int) * (nv + 1));
        fread(v_pos, sizeof(int), nv + 1, file);
        fclose(file);

        file = fopen(argv[2], head ? "w" : "a");
        if(file == NULL) {
            printf("Failed to open %s\n", argv[2]);
        }
        str = (char*)malloc(sizeof(char)*(strlen(argv[1])+1));
        strcpy(str, argv[1]);
        write_stats(file, find_name(str), nv, v_pos, head);
        free(str);
        free(v_pos);
        fclose(file);
        return 0;
    } else {
        printf("USAGE: %s <input file> <output file>\n", argv[0]);
        return 1;
    }
}
