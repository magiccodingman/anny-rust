#include "anny.h"
#include <math.h>
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
    anny_model_f32_free(s);
    anny_model_free(m);
    /* The output remains readable after both models are freed. */
    printf("C f32: %zu vertices; max difference %.9g; first x %.9g\n",bv.len/3,largest,bv.data[0]);
    anny_output_free(a);
    anny_output_f32_free(b);
    anny_model_f32_free(copy);
    return 0;
}
