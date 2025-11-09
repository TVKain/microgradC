#include "nn.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>    

#define READ_OR_FAIL(ptr, type, count, file, onfail) \
    do { \
        if (fread((ptr), sizeof(type), (size_t)(count), (file)) != (size_t)(count)) { \
            fprintf(stderr, "Read failed at %s:%d\n", __FILE__, __LINE__); \
            onfail; \
        } \
    } while (0)

uint32_t read_be_uint32(FILE *f) {
    uint8_t b[4];
    READ_OR_FAIL(b, uint8_t, 4, f, { return 0; });
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

unsigned char **load_mnist_images(Arena *arena, const char *filename, int *num_images, int *rows, int *cols) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror(filename); return NULL; }

    uint32_t magic = read_be_uint32(f);
    if (magic != 2051) { fprintf(stderr, "Invalid MNIST image file: %s\n", filename); fclose(f); return NULL; }

    *num_images = read_be_uint32(f);
    *rows = read_be_uint32(f);
    *cols = read_be_uint32(f);

    unsigned char **images = arena_alloc(arena, *num_images * sizeof(unsigned char*));
    for (int i = 0; i < *num_images; i++) {
        images[i] = arena_alloc(arena, (*rows) * (*cols));
        READ_OR_FAIL(images[i], unsigned char, (*rows) * (*cols), f, { fclose(f); return NULL; });
    }

    fclose(f);
    return images;
}

unsigned char *load_mnist_labels(Arena *arena, const char *filename, int *num_labels) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror(filename); return NULL; }

    uint32_t magic = read_be_uint32(f);
    if (magic != 2049) { fprintf(stderr, "Invalid MNIST label file: %s\n", filename); fclose(f); return NULL; }

    *num_labels = read_be_uint32(f);

    unsigned char *labels = arena_alloc(arena, *num_labels);
    READ_OR_FAIL(labels, unsigned char, *num_labels, f, { fclose(f); return NULL; });

    fclose(f);
    return labels;
}

void render_image(unsigned char *img, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            unsigned char px = img[r * cols + c];
            // Map pixel to ASCII: light = '.', dark = '#'
            if (px > 128) printf("#");
            else printf(".");
        }
        printf("\n");
    }
}

Value **image_to_value(Arena *a, unsigned char *image, int size) {
    Value **out = arena_alloc(a, sizeof(Value*) * size);
    
    for (int i = 0; i < size; ++i) {
        out[i] = value_alloc(a, (double)image[i] / 255.0);
    }

    return out;
}

Value *label_to_value(Arena *a, unsigned char label) {
    return value_alloc(a, (double)label);
}

int main() {
    Arena mnist_arena = {0};
    Arena param_arena = {0};
    Arena graph_arena = {0};

    const char *image_file = "mnist/train-images.idx3-ubyte";
    const char *label_file = "mnist/train-labels.idx1-ubyte";

    int num_images, rows, cols;
    
    unsigned char **images = load_mnist_images(&mnist_arena, image_file, &num_images, &rows, &cols);

    int size = rows * cols;
    if (!images) return 1;

    int num_labels;
    unsigned char *labels = load_mnist_labels(&mnist_arena, label_file, &num_labels);
    if (!labels) return 1;

    printf("Loaded %d images of size %dx%d\n", num_images, rows, cols);
    printf("Loaded %d labels\n", num_labels);

    Layer_Config cfgs[2] = {
        NN_LAYER_CFG(784, 8, ACT_RELU),
        NN_LAYER_CFG(8, 10, ACT_LINEAR)
    };

    MLP *mlp = mlp_alloc(&param_arena, cfgs, 2);

    int epochs = 50;
    double lr = 0.04;
    int sample_size = 100;  // number of images per epoch

    srand((unsigned int)time(NULL)); // seed RNG

    for (int epoch = 0; epoch < epochs; ++epoch) {
        printf("Epoch: %d\n", epoch);
        double total_loss = 0;

        for (int i = 0; i < sample_size; ++i) {
            int idx = rand() % num_images;  // random image index

            Value **image = image_to_value(&graph_arena, images[idx], size);
            Value *target = label_to_value(&graph_arena, labels[idx]);

            // Forward pass 
            Value **out = mlp_forward(&graph_arena, mlp, image, (size_t)size);

            // Loss
            Value *loss = cross_entropy(&graph_arena, out, target, 10);
            total_loss += loss->data;

            // Backprop
            value_backward(&graph_arena, loss);

            // Update
            mlp_update(mlp, lr);

            // Zero grad
            mlp_zero_grad(mlp);

            // Reset graph arena for next sample
            arena_reset(&graph_arena);
        }

        printf("Avg Loss: %.4f\n", total_loss / sample_size);
    }

    mlp_save(mlp, "mnist.bin");

    arena_free(&graph_arena);
    arena_free(&param_arena);
    arena_free(&mnist_arena);
    return 0;
}
