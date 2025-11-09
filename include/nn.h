#ifndef NN_H
#define NN_H

#include "arena.h"
#include "value.h"

#include <time.h>
#include <stdlib.h>
#include <string.h>


typedef struct Neuron Neuron;

typedef enum {
    ACT_TANH 
} Act_Kind;

/*
 * (n_in, 1)
 * y = act_fn(sum(wi * xi) + b), i = 0 to n_in
 */
struct Neuron {
    Value **ws; 
    Value *b; 
    size_t n_in;
};

Neuron *neuron_alloc(Arena *a, size_t n_in);
void neuron_print(Neuron *n);
void neuron_zero_grad(Neuron *n);

typedef struct Layer Layer;

/*
 * (n_in, n_out)
 */
struct Layer {
    Neuron **neurons;
    size_t n_in;
    size_t n_out;
};

Layer *layer_alloc(Arena *a, size_t n_in, size_t n_out);
void layer_print(Layer *l);
void layer_zero_grad(Layer *l);

typedef struct MLP MLP;

struct MLP {
    Layer **layers;
    size_t *dims;
    size_t dim_size;
};

MLP *mlp_alloc(Arena *a, size_t *dims, size_t dim_size);
void mlp_print(MLP *m);
Value **mlp_forward(Arena *a, MLP *m, Value **x, size_t x_size);
void mlp_zero_grad(MLP *m);
void mlp_update(MLP *m, double lr);
Value *mse(Arena *a, Value *pred, Value *target);

#endif