# MicrogradC
- [Micrograd](https://github.com/karpathy/micrograd) in C
- I built this to understand how neural networks actually work under the hood 
- Uses Arena Allocator from [Tsoding](https://github.com/tsoding) for memory management

# Examples
```C
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
```
```
Epoch    0, Avg Loss: 1.132206
Epoch  500, Avg Loss: 0.136454
Epoch 1000, Avg Loss: 0.001060
Epoch 1500, Avg Loss: 0.000429
Epoch 2000, Avg Loss: 0.000261
Epoch 2500, Avg Loss: 0.000186
Epoch 3000, Avg Loss: 0.000143
Epoch 3500, Avg Loss: 0.000116
Epoch 4000, Avg Loss: 0.000097
Epoch 4500, Avg Loss: 0.000084

--- Final Results ---
Input: [0, 0], Target: 0, Prediction: 0.0002
Input: [0, 1], Target: 1, Prediction: 0.9878
Input: [1, 0], Target: 1, Prediction: 0.9880
Input: [1, 1], Target: 0, Prediction: -0.0001
```

# References
- https://github.com/karpathy/micrograd
- https://github.com/tsoding/arena
