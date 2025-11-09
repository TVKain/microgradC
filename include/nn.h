#ifndef NN_H
#define NN_H

#include "arena.h"
#include "value.h"

#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define LAYER_CFG(n_in_val, n_out_val, act_val) \
    ((Layer_Config){ .n_in = (n_in_val), .n_out = (n_out_val), .act = (act_val) })

typedef struct Neuron Neuron;

/* Activation kinds used by neurons or activation layers */
typedef enum {
    ACT_LINEAR,
    ACT_TANH,
    ACT_RELU,
    ACT_SIGMOID
} Act_Kind;

/*
 * (n_in, 1)
 * y = act_fn(sum(wi * xi) + b), i = 0..n_in-1
 */
struct Neuron {
    Value **ws;     /* array of Value* (weights) length n_in */
    Value *b;       /* bias */
    size_t n_in;
    Act_Kind act;
};

Neuron *neuron_alloc(Arena *a, size_t n_in, Act_Kind act);
void neuron_print(Neuron *n);
void neuron_zero_grad(Neuron *n);

typedef struct Layer Layer;

/* (n_in, n_out) for linear, for activation layers n_in == n_out */
struct Layer {
    Act_Kind act;
    size_t n_in;
    size_t n_out;

    /* only used for linear layers */
    Neuron **neurons;
};

typedef struct Layer_Config Layer_Config;
struct Layer_Config {
    size_t n_in;
    size_t n_out;
    Act_Kind act;    
};

Layer *layer_alloc(Arena *a, Layer_Config *cfg);
void layer_print(Layer *l);
void layer_zero_grad(Layer *l);

/* MLP */
typedef struct MLP MLP;
struct MLP {
    Layer **layers;
    size_t layer_size;
};

MLP *mlp_alloc(Arena *a, Layer_Config *layer_configs, size_t config_size);
void mlp_print(MLP *m);
Value **mlp_forward(Arena *a, MLP *m, Value **x, size_t x_size);
void mlp_zero_grad(MLP *m);
void mlp_update(MLP *m, double lr);

#endif
