#include "anny.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#define CHECK(expr) do { if ((expr) != 0) { fprintf(stderr,"%s\n",anny_last_error()); return 1; } } while (0)
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    AnnyModel *m = NULL;
    AnnyModelF32 *s = NULL, *copy = NULL;
    AnnyOutput *a = NULL;
    AnnyOutputF32 *b = NULL;
    AnnyBytes *bytes = NULL;
    AnnyTensorView av;
    AnnyTensorViewF32 bv;
    CHECK(anny_model_build(argv[1],NULL,&m));
    CHECK(anny_model_to_f32(m,&s));
    CHECK(anny_model_evaluate(m,NULL,&a));
    CHECK(anny_model_f32_evaluate(s,NULL,&b));
    CHECK(anny_output_tensor(a,"vertices",&av));
    CHECK(anny_output_f32_tensor(b,"vertices",&bv));
    if (av.len != bv.len || bv.len == 0) return 3;
    double largest = 0;
    for (size_t i=0;i<av.len;i++) {
        double e=fabs(av.data[i]-bv.data[i]);
        if (!isfinite(bv.data[i]) || e>2e-4) return 4;
        if (e>largest) largest=e;
    }
    CHECK(anny_model_f32_prepared_bytes(s,&bytes));
    CHECK(anny_model_f32_from_bytes(anny_bytes_data(bytes),anny_bytes_len(bytes),NULL,&copy));
    anny_bytes_free(bytes);
    /* f32 pose session: the pose-only update path must agree with the f32 evaluate bit for bit. */
    const char *pose =
        "{\"root\":[[1,0,0,0.2],[0,0.9396926,0.3420201,-0.15],[0,-0.3420201,0.9396926,0.05],[0,0,0,1]]}";
    char parameters[512];
    snprintf(parameters,sizeof parameters,"{\"pose_parameters\":%s}",pose);
    AnnyOutputF32 *e = NULL;
    AnnySessionF32 *session = NULL;
    CHECK(anny_model_f32_evaluate(s,parameters,&e));
    CHECK(anny_session_f32_new(s,parameters,&session));
    CHECK(anny_session_f32_update(session,pose));
    AnnyTensorViewF32 from_session, from_evaluate;
    CHECK(anny_session_f32_tensor(session,"vertices",&from_session));
    CHECK(anny_output_f32_tensor(e,"vertices",&from_evaluate));
    if (from_session.len != from_evaluate.len || from_session.len == 0) return 5;
    if (memcmp(from_session.data,from_evaluate.data,from_session.len*sizeof(float)) != 0) return 6;
    float *snapshot = malloc(from_session.len*sizeof(float));
    if (snapshot == NULL) return 7;
    memcpy(snapshot,from_session.data,from_session.len*sizeof(float));
    size_t snapshot_len = from_session.len;
    anny_output_f32_free(e);
    /* The session keeps its own reference to the model, so the handle may be freed first. */
    anny_model_f32_free(s);
    CHECK(anny_session_f32_update(session,pose));
    CHECK(anny_session_f32_tensor(session,"vertices",&from_session));
    if (from_session.len != snapshot_len ||
        memcmp(snapshot,from_session.data,snapshot_len*sizeof(float)) != 0) return 8;
    free(snapshot);
    anny_session_f32_free(session);
    puts("C f32 pose session: pose-only updates match f32 evaluate exactly and outlive the model handle");
    anny_model_free(m);
    /* The output remains readable after both models are freed. */
    printf("C f32: %zu vertices; max difference %.9g; first x %.9g\n",bv.len/3,largest,bv.data[0]);
    anny_output_free(a);
    anny_output_f32_free(b);
    anny_model_f32_free(copy);
    return 0;
}
