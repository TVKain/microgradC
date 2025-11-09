#include "value.h"
#include "nn.h"

int main(void) {
    Arena param_arena = {0};
    Arena graph_arena = {0};

    MLP *mlp = mlp_load(&param_arena, "xor.bin");

    // XOR dataset
    double X[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    double y[4] = {0.0, 1.0, 1.0, 0.0};

    for (int i = 0; i < 4; i++) {

        Value *inputs[2];
        inputs[0] = value_alloc(&graph_arena, X[i][0]);
        inputs[1] = value_alloc(&graph_arena, X[i][1]);

        Value **out = mlp_forward(&graph_arena, mlp, inputs, 2);
        printf("Input: [%.0f, %.0f] | Target: %.0f | Pred: %.4f\n",
               X[i][0], X[i][1], y[i], out[0]->data);
        arena_reset(&graph_arena);
    }

    mlp_print(mlp);

    arena_free(&graph_arena);
    arena_free(&param_arena);
    return 0;
}
