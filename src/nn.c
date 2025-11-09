#include "nn.h"
#include "value.h"

#include <math.h>
#include <time.h>

static double rand_from(double min, double max) {
    double u = (double)rand() / (double)RAND_MAX;
    return min + u * (max - min);
}


/* Neuron */
Neuron *neuron_alloc(Arena *a, size_t n_in, Act_Kind act) {
    Neuron *n = arena_alloc(a, sizeof(Neuron));
    n->n_in = n_in;
    n->act = act;

    /* allocate array of Value* for weights */
    n->ws = arena_alloc(a, sizeof(Value*) * n_in);
    for (size_t i = 0; i < n_in; ++i) {
        n->ws[i] = value_alloc(a, rand_from(-1, 1));
    }

    n->b = value_alloc(a, rand_from(-1, 1));
    return n;
}

void neuron_print(Neuron *n) {
    printf("\tNeuron(n_in=%zu act=%d) ", n->n_in, (int)n->act);
    for (size_t i = 0; i < n->n_in; ++i) {
        printf("w[%zu]=%.4f ", i, n->ws[i]->data);
    }
    printf("b=%.4f", n->b->data);
}

void neuron_zero_grad(Neuron *n) {
    for (size_t i = 0; i < n->n_in; ++i) {
        n->ws[i]->grad = 0.0;
    }
    n->b->grad = 0.0;
}

/* Layer */
Layer *layer_alloc(Arena *a, Layer_Config *cfg) {
    Layer *layer = arena_alloc(a, sizeof(Layer));
    layer->n_in = cfg->n_in;
    layer->n_out = cfg->n_out;

    
    layer->neurons = arena_alloc(a, sizeof(Neuron*) * cfg->n_out);
    for (size_t i = 0; i < cfg->n_out; ++i) {
        layer->neurons[i] = neuron_alloc(a, cfg->n_in, cfg->act);
    }
    
    return layer;
}

void layer_print(Layer *l) {
    printf("Layer(kind=%d in=%zu out=%zu)\n", l->n_in, l->n_out);

    for (size_t i = 0; i < l->n_out; ++i) {
        printf("\t");
        neuron_print(l->neurons[i]);
        printf("\n");
    }
}

void layer_zero_grad(Layer *l) {
    for (size_t i = 0; i < l->n_out; ++i) {
        neuron_zero_grad(l->neurons[i]);
    }
}


/* Forward */
static Value *neuron_forward(Arena *a, Neuron *n, Value **x, size_t x_size) {
    if (n->n_in != x_size) {
        fprintf(stderr, "neuron_forward: invalid dimension (expect %zu got %zu)\n", n->n_in, x_size);
        exit(1);
    }

    /* Visualize purposes */
    for (size_t i = 0; i < x_size; ++i) {
        x[i]->value_kind = VALUE_INPUT;
    }

    Value *out = value_alloc(a, 0.0);
    out->value_kind = VALUE_BOOTSTRAP;

    for (size_t i = 0; i < x_size; ++i) {
        Value *w = n->ws[i];
        w->value_kind = VALUE_PARAM;
        Value *mul = value_mul(a, w, x[i]);
        out = value_add(a, out, mul);
    }

    /* Add bias */
    n->b->value_kind = VALUE_PARAM;
    out = value_add(a, out, n->b);

    /* Activation */
    switch (n->act) {
        case ACT_TANH:    out = value_tanh(a, out); break;
        case ACT_RELU:    out = value_relu(a, out); break;
        case ACT_SIGMOID: out = value_sigmoid(a, out); break;
        case ACT_LINEAR:
        default: break;
    }

    return out;
}


/* Layer forward */
Value **layer_forward(Arena *a, Layer *l, Value **x, size_t x_size) {
    if (l->n_in != x_size) {
        fprintf(stderr, "layer_forward: invalid dimension (expect %zu got %zu)\n", l->n_in, x_size);
        exit(1);
    }

    Value **out = arena_alloc(a, sizeof(Value*) * l->n_out);
    for (size_t i = 0; i < l->n_out; ++i) {
        out[i] = neuron_forward(a, l->neurons[i], x, x_size);
    }
    return out;
}

// MLP

MLP *mlp_alloc(Arena *a, Layer_Config *layer_configs, size_t config_size) {
    MLP *mlp = arena_alloc(a, sizeof(MLP));
    mlp->layer_size = config_size;
    mlp->layers = arena_alloc(a, sizeof(Layer*) * config_size);

    for (size_t i = 0; i < config_size; ++i) {
        mlp->layers[i] = layer_alloc(a, &layer_configs[i]);
    }
    return mlp;
}

void mlp_print(MLP *m) {
    printf("MLP (layers=%zu)\n", m->layer_size);
    for (size_t i = 0; i < m->layer_size; ++i) {
        printf("\t[%zu] ", i);
        layer_print(m->layers[i]);
    }
}

/* Build the DAG: pass the vector through every layer */
Value **mlp_forward(Arena *a, MLP *m, Value **x, size_t x_size) {
    if (m->layer_size == 0) return x;

    if (m->layers[0]->n_in != x_size) {
        fprintf(stderr, "mlp_forward: input size mismatch (expect %zu got %zu)\n", m->layers[0]->n_in, x_size);
        exit(1);
    }

    Value **out = x;
    size_t current_size = x_size;

    for (size_t i = 0; i < m->layer_size; ++i) {
        out = layer_forward(a, m->layers[i], out, current_size);
        current_size = m->layers[i]->n_out;
    }
    return out;
}

/* zero grads */
void mlp_zero_grad(MLP *m) {
    for (size_t i = 0; i < m->layer_size; ++i) {
        layer_zero_grad(m->layers[i]);
    }
}

static void neuron_update(Neuron *n, double lr) {
    for (size_t i = 0; i < n->n_in; ++i) {
        n->ws[i]->data -= lr * n->ws[i]->grad;
    }
    n->b->data -= lr * n->b->grad;
}

static void layer_update(Layer *l, double lr) {
    for (size_t i = 0; i < l->n_out; ++i) {
        neuron_update(l->neurons[i], lr);
    }
}

void mlp_update(MLP *m, double lr) {
    for (size_t i = 0; i < m->layer_size; ++i) {
        layer_update(m->layers[i], lr);
    }
}
