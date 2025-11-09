# MicrogradC
- [Micrograd](https://github.com/karpathy/micrograd) in C
- I built this to understand how neural networks are implemented under the hood 
- Uses Arena Allocator from [Tsoding](https://github.com/tsoding) for memory management

## Requirements
- C compiler `gcc`, `clang`, ...
- `make`

## Examples

### XOR with Cross-Entropy
```C
#include "arena.h"
#include "value.h"
#include "nn.h"

int main() {
    Arena param_arena = {0};
    Arena graph_arena = {0};

    // Define MLP with 2 inputs, 16 hidden, 2 outputs
    Layer_Config cfgs[2] = {
        LAYER_CFG(2, 16, ACT_TANH),   // hidden
        LAYER_CFG(16, 2, ACT_LINEAR)  // output
    };

    MLP *mlp = mlp_alloc(&param_arena, cfgs, 2);

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
```
```
Epoch    0 | Avg Loss: 0.991271
Epoch  500 | Avg Loss: 0.376778
Epoch 1000 | Avg Loss: 0.189224
Epoch 1500 | Avg Loss: 0.254065
Epoch 2000 | Avg Loss: 0.025508
Epoch 2500 | Avg Loss: 0.245966
Epoch 3000 | Avg Loss: 0.034588
Epoch 3500 | Avg Loss: 0.066744
Epoch 4000 | Avg Loss: 0.001242
Epoch 4500 | Avg Loss: 0.000742

--- Final Results ---
Input: [0, 0] | Target: 0 | Output: (61.4917, 53.3319) | Softmax: (0.9997, 0.0003)
Input: [0, 1] | Target: 1 | Output: (-39.1342, -30.7677) | Softmax: (0.0002, 0.9998)
Input: [1, 0] | Target: 1 | Output: (-40.8965, -33.7789) | Softmax: (0.0008, 0.9992)
Input: [1, 1] | Target: 0 | Output: (-49.6678, -56.9846) | Softmax: (0.9993, 0.0007)
```

## Running examples
```bash
make run/<example-name>
```

### Running mnist
```bash
make run/mnist      # train and save
make run/mnist_eval # evaluation
```
```bash
Running example: mnist_eval
Evaluation on 10000 random test images:
Pre-trained MLP Accuracy: 21.70% (2170/10000)
Random MLP Accuracy:     5.05% (505/10000)
```

## Note
- MicrogradC is very slow for MNIST, especially with larger models. For shits and giggles only.

## References
- https://github.com/karpathy/micrograd
- https://github.com/tsoding/arena
