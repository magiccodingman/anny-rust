#include "anny.h"
#include <stdio.h>
#include <stdlib.h>
static void check(int status) {
    if (status) { fprintf(stderr, "Anny: %s\n", anny_last_error()); exit(1); }
}
int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: c_smoke /path/to/imported/data\n"); return 2; }
    AnnyModel *model = NULL;
    AnnyOutput *output = NULL;
    check(anny_model_build(argv[1], NULL, &model));
    check(anny_model_evaluate(model, "{\"phenotype_kwargs\":{\"height\":0.6}}", &output));
    AnnyTensorView vertices, faces;
    check(anny_output_tensor(output, "vertices", &vertices));
    check(anny_model_tensor(model, "faces", &faces));
    printf("ABI %u: %zu vertices, %zu faces; first vertex %.9f %.9f %.9f\n",
        anny_abi_version(), vertices.shape[1], faces.shape[0], vertices.data[0], vertices.data[1], vertices.data[2]);
    anny_output_free(output);
    anny_model_free(model);
    return 0;
}
