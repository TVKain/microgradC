#include "arena.h"
#include "value.h"
#include "nn.h"

int main() {
    Arena param_arena = {0};
    Arena graph_arena = {0};

    // Define MLP with 2 inputs, 16 hidden, 2 outputs
    Layer_Config cfgs[2] = {
        NN_LAYER_CFG(2, 16, ACT_TANH),   // hidden
        NN_LAYER_CFG(16, 2, ACT_LINEAR)  // output
    };

    MLP *mlp = mlp_alloc(&param_arena, cfgs, 2);
    mlp_print(mlp);
    printf("\n");

    // XOR dataset (2-class)
    double X[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    double Y[4] = {0, 1, 1, 0};

    int num_epochs = 5000;
    double learning_rate = 0.1;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < 4; i++) {
            arena_reset(&graph_arena);

            // Inputs
            Value *inputs[2];
            inputs[0] = value_alloc(&graph_arena, X[i][0]);
            inputs[1] = value_alloc(&graph_arena, X[i][1]);
            Value *target = value_alloc(&graph_arena, Y[i]);

            // Forward
            Value **out = mlp_forward(&graph_arena, mlp, inputs, 2);
            Value *loss = cross_entropy(&graph_arena, out, target, 2);

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
        Value **probs = soft_max(&graph_arena, out, 2);

        printf("Input: [%.0f, %.0f] | Target: %.0f | Output: (%.4f, %.4f) | Softmax: (%.4f, %.4f)\n",
               X[i][0], X[i][1], Y[i], out[0]->data, out[1]->data,
               probs[0]->data, probs[1]->data);
    }

    arena_free(&graph_arena);
    arena_free(&param_arena);
    return 0;
}
