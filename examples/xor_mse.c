#include "arena.h"
#include "value.h"
#include "nn.h"

int main(void) {
    Arena param_arena = {0};
    Arena graph_arena = {0};

    // Define network architecture with Layer_Config
    Layer_Config cfgs[2] = {
        LAYER_CFG(2, 2, ACT_TANH),
        LAYER_CFG(2, 1, ACT_LINEAR)
    };

    MLP *mlp = mlp_alloc(&param_arena, cfgs, 2);
    mlp_print(mlp);
    printf("\n");

    // XOR dataset
    double X[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    double y[4] = {0.0, 1.0, 1.0, 0.0};

    int num_epochs = 1000;
    double learning_rate = 0.1;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < 4; i++) {
            arena_reset(&graph_arena);

            // Prepare inputs
            Value *inputs[2];
            inputs[0] = value_alloc(&graph_arena, X[i][0]);
            inputs[1] = value_alloc(&graph_arena, X[i][1]);
            inputs[0]->value_kind = VALUE_INPUT;
            inputs[1]->value_kind = VALUE_INPUT;

            // Prepare target
            Value *target = value_alloc(&graph_arena, y[i]);
            Value *targets[1] = { target };

            // Forward
            Value **out = mlp_forward(&graph_arena, mlp, inputs, 2);
            Value *loss = mse(&graph_arena, out, targets, 1);

            total_loss += loss->data;

            // Backward
            value_backward(&graph_arena, loss);

            // Update parameters
            mlp_update(mlp, learning_rate);
            mlp_zero_grad(mlp);
        }

        if (epoch % 500 == 0)
            printf("Epoch %4d | Avg Loss: %.6f\n", epoch, total_loss / 4.0);
    }

    printf("\n--- Final Results ---\n");
    for (int i = 0; i < 4; i++) {
        arena_reset(&graph_arena);

        Value *inputs[2];
        inputs[0] = value_alloc(&graph_arena, X[i][0]);
        inputs[1] = value_alloc(&graph_arena, X[i][1]);

        Value **out = mlp_forward(&graph_arena, mlp, inputs, 2);
        printf("Input: [%.0f, %.0f] | Target: %.0f | Pred: %.4f\n",
               X[i][0], X[i][1], y[i], out[0]->data);
    }

    arena_free(&graph_arena);
    arena_free(&param_arena);
    return 0;
}
