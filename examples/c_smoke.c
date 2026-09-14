#include "anny.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void check(int status) {
    if (status) { fprintf(stderr, "Anny: %s\n", anny_last_error()); exit(1); }
}
static void require(int condition, const char *message) {
    if (!condition) { fprintf(stderr, "%s\n", message); exit(1); }
}
int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: c_smoke /path/to/imported/data\n"); return 2; }
    AnnyModel *model = NULL, *transformed = NULL, *reloaded = NULL;
    AnnyOutput *output = NULL;
    check(anny_model_build_cached(argv[1], NULL, NULL, &model));
    check(anny_model_evaluate(model, "{\"phenotype_kwargs\":{\"height\":0.6}}", &output));
    AnnyTensorView vertices, faces;
    check(anny_output_tensor(output, "vertices", &vertices));
    check(anny_model_tensor(model, "faces", &faces));
    require(vertices.rank == 3 && vertices.shape[1] > 0, "empty vertices");
    require(faces.rank == 2 && faces.shape[0] > 0, "empty faces");
    printf("ABI %u: %zu vertices, %zu faces; first vertex %.9f %.9f %.9f\n",
        anny_abi_version(), vertices.shape[1], faces.shape[0], vertices.data[0], vertices.data[1], vertices.data[2]);
    anny_output_free(output);
    char *text = NULL;
    check(anny_model_query(model, "{\"operation\":\"measure\"}", &text));
    require(text && strstr(text, "waist_circumference"), "measurement response missing");
    anny_string_free(text);
    AnnyBytes *glb = NULL, *prepared = NULL;
    check(anny_model_export_glb(model, NULL, "{\"rigged\":true}", &glb));
    require(anny_bytes_len(glb) > 12 && memcmp(anny_bytes_data(glb), "glTF", 4) == 0, "invalid GLB header");
    anny_bytes_free(glb);
    check(anny_model_transform(model, "[{\"op\":\"triangulate\"},{\"op\":\"compact-skinning-weights\"}]", &transformed));
    check(anny_model_transfer_pose(model, transformed, NULL, "local-ref", &text));
    require(text && strstr(text, "pose_parameters"), "pose transfer response missing");
    anny_string_free(text);
    check(anny_model_prepared_bytes(transformed, &prepared));
    anny_model_free(transformed);
    anny_model_free(model);
    check(anny_model_from_bytes(anny_bytes_data(prepared), anny_bytes_len(prepared), NULL, &reloaded));
    anny_bytes_free(prepared);
    check(anny_model_evaluate(reloaded, NULL, &output));
    check(anny_output_tensor(output, "vertices", &vertices));
    require(vertices.shape[1] > 0, "prepared reload returned no vertices");
    anny_output_free(output);
    anny_model_free(reloaded);
    puts("C query, rigged GLB, transform, pose transfer, owned bytes and reload: PASS");
    return 0;
}
