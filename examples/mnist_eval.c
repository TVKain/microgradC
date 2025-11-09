#include "nn.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

uint32_t read_be_uint32(FILE *f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

unsigned char **load_mnist_images(Arena *arena, const char *filename, int *num_images, int *rows, int *cols) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror(filename); return NULL; }

    if (read_be_uint32(f) != 2051) { fprintf(stderr, "Invalid MNIST image file\n"); fclose(f); return NULL; }
    *num_images = read_be_uint32(f);
    *rows = read_be_uint32(f);
    *cols = read_be_uint32(f);

    unsigned char **images = arena_alloc(arena, *num_images * sizeof(unsigned char*));
    for (int i = 0; i < *num_images; ++i) {
        images[i] = arena_alloc(arena, (*rows) * (*cols));
        if (fread(images[i], 1, (*rows) * (*cols), f) != (size_t)((*rows) * (*cols))) {
            fprintf(stderr, "Failed to read image %d\n", i);
            fclose(f);
            return NULL;
        }
    }

    fclose(f);
    return images;
}

unsigned char *load_mnist_labels(Arena *arena, const char *filename, int *num_labels) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror(filename); return NULL; }

    if (read_be_uint32(f) != 2049) { fprintf(stderr, "Invalid MNIST label file\n"); fclose(f); return NULL; }
    *num_labels = read_be_uint32(f);

    unsigned char *labels = arena_alloc(arena, *num_labels);
    if (fread(labels, 1, *num_labels, f) != (size_t)(*num_labels)) {
        fprintf(stderr, "Failed to read labels\n");
        fclose(f);
        return NULL;
    }

    fclose(f);
    return labels;
}

Value **image_to_value(Arena *a, unsigned char *image, int size) {
    Value **out = arena_alloc(a, sizeof(Value*) * size);
    for (int i = 0; i < size; ++i)
        out[i] = value_alloc(a, (double)image[i] / 255.0);
    return out;
}

int evaluate_mlp(MLP *mlp, unsigned char **images, unsigned char *labels, int num_images, int rows, int cols, int sample_count) {
    Arena graph_arena = {0};
    int size = rows * cols;
    int correct = 0;
    srand((unsigned int)time(NULL));

    for (int i = 0; i < sample_count; ++i) {
        int idx = rand() % num_images;

        Value **image = image_to_value(&graph_arena, images[idx], size);

        // Forward pass
        Value **out = mlp_forward(&graph_arena, mlp, image, size);

        // Find predicted label
        int pred = 0;
        double max_val = out[0]->data;
        for (int j = 1; j < 10; ++j) {
            if (out[j]->data > max_val) {
                max_val = out[j]->data;
                pred = j;
            }
        }

        if (pred == labels[idx]) ++correct;

        arena_reset(&graph_arena);
    }

    arena_free(&graph_arena);
    return correct;
}

int main() {
    Arena mnist_arena = {0};

    const char *test_images_file = "mnist/t10k-images.idx3-ubyte";
    const char *test_labels_file = "mnist/t10k-labels.idx1-ubyte";

    int num_images, rows, cols;
    unsigned char **images = load_mnist_images(&mnist_arena, test_images_file, &num_images, &rows, &cols);
    if (!images) return 1;

    int num_labels;
    unsigned char *labels = load_mnist_labels(&mnist_arena, test_labels_file, &num_labels);
    if (!labels) return 1;

    // Load pre-trained MLP
    MLP *mlp_trained = mlp_load(&mnist_arena, "mnist.bin");
    if (!mlp_trained) { fprintf(stderr, "Failed to load trained model\n"); return 1; }

    // Create a random MLP with same architecture
    Layer_Config cfgs[2] = {
        NN_LAYER_CFG(784, 8, ACT_RELU),
        NN_LAYER_CFG(8, 10, ACT_LINEAR)
    };
    MLP *mlp_random = mlp_alloc(&mnist_arena, cfgs, 2);

    int sample_count = 10000;

    int correct_trained = evaluate_mlp(mlp_trained, images, labels, num_images, rows, cols, sample_count);
    int correct_random = evaluate_mlp(mlp_random, images, labels, num_images, rows, cols, sample_count);

    printf("Evaluation on %d random test images:\n", sample_count);
    printf("Pre-trained MLP Accuracy: %.2f%% (%d/%d)\n", 100.0 * correct_trained / sample_count, correct_trained, sample_count);
    printf("Random MLP Accuracy:     %.2f%% (%d/%d)\n", 100.0 * correct_random / sample_count, correct_random, sample_count);

    arena_free(&mnist_arena);
    return 0;
}
