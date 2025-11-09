#include "arena.h"
#include "value.h"
#include "nn.h"

int main() {
    Arena param_arena = {0};
    Arena graph_arena = {0};

    size_t dims[3] = {2, 2, 1};
    MLP *mlp = mlp_alloc(&param_arena, dims, 3);

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

    int num_epochs = 5000;
    double learning_rate = 0.05;

    // Training loop, currently SGD
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double total_loss = 0.0;
        for (int i = 0; i < 4; i++) {
            Value *inputs[2];
            inputs[0] = value_create(&graph_arena, X[i][0]);
            inputs[1] = value_create(&graph_arena, X[i][1]);
            inputs[0]->value_kind = VALUE_INPUT;
            inputs[1]->value_kind = VALUE_INPUT;
            
            // Create target Value
            Value *target = value_create(&graph_arena, y[i]);
            target->value_kind = VALUE_INPUT;
            
            // Forward pass
            Value **out = mlp_forward(&graph_arena, mlp, inputs, 2);
            Value *loss = mse(&graph_arena, out[0], target);
            
            total_loss += loss->data;
            
            // Backward pass
            value_backward(&graph_arena, loss);
            
            // Update parameters
            mlp_update(mlp, learning_rate);
            
            // Zero gradients
            mlp_zero_grad(mlp);
        }

        if (epoch % 500 == 0) {
            printf("Epoch %4d, Avg Loss: %.6f\n", epoch, total_loss / 4.0);
        }
    }

    printf("\n--- Final Results ---\n");
    for (int i = 0; i < 4; i++) {
        Value *inputs[2];
        inputs[0] = value_create(&graph_arena, X[i][0]);
        inputs[1] = value_create(&graph_arena, X[i][1]);
        
        Value **out = mlp_forward(&graph_arena, mlp, inputs, 2);
        
        printf("Input: [%.0f, %.0f], Target: %.0f, Prediction: %.4f\n",
               X[i][0], X[i][1], y[i], out[0]->data);
        
        arena_reset(&graph_arena);
    }

    arena_free(&graph_arena);
    arena_free(&param_arena);
    return 0;
}
