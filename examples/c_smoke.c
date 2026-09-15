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
    /* Pose session: the pose-only update path must agree with evaluate bit for bit. */
    const char *pose =
        "{\"root\":[[1,0,0,0.2],[0,0.9396926,0.3420201,-0.15],[0,-0.3420201,0.9396926,0.05],[0,0,0,1]]}";
    char parameters[512];
    snprintf(parameters, sizeof parameters, "{\"pose_parameters\":%s}", pose);
    AnnyOutput *evaluated = NULL;
    AnnySession *session = NULL;
    check(anny_model_evaluate(reloaded, parameters, &evaluated));
    check(anny_session_new(reloaded, parameters, &session));
    check(anny_session_update(session, pose));
    AnnyTensorView from_session, from_evaluate;
    check(anny_session_tensor(session, "vertices", &from_session));
    check(anny_output_tensor(evaluated, "vertices", &from_evaluate));
    require(from_session.len == from_evaluate.len && from_session.len > 0, "session/evaluate length mismatch");
    require(memcmp(from_session.data, from_evaluate.data, from_session.len * sizeof(double)) == 0,
        "session output differs from evaluate");
    double *snapshot = malloc(from_session.len * sizeof(double));
    require(snapshot != NULL, "out of memory");
    memcpy(snapshot, from_session.data, from_session.len * sizeof(double));
    size_t snapshot_len = from_session.len;
    anny_output_free(evaluated);
    /* The session keeps its own reference to the model, so the handle may be freed first. */
    anny_model_free(reloaded);
    check(anny_session_update(session, pose));
    check(anny_session_tensor(session, "vertices", &from_session));
    require(from_session.len == snapshot_len
        && memcmp(snapshot, from_session.data, snapshot_len * sizeof(double)) == 0,
        "session lost its model or changed its output");
    free(snapshot);
    anny_session_free(session);
    puts("C query, rigged GLB, transform, pose transfer, owned bytes, reload and pose session: PASS");
    return 0;
}
