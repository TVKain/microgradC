#include "nn.h"
#include "value.h"

double rand_from(double min, double max) {
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

Neuron *neuron_alloc(Arena *a, size_t n_in) {
    Neuron *n = arena_alloc(a, sizeof(Neuron));

    n->ws = arena_alloc(a, sizeof(double*) * n_in);

    for (size_t i = 0; i < n_in; ++i) {
        n->ws[i] = value_create(a, rand_from(-1, 1));
    }

    n->b = value_create(a, rand_from(-1, 1));

    n->n_in = n_in;

    return n;
}

void neuron_print(Neuron *n) {
    printf("Neuron ");
    for (size_t i = 0; i < n->n_in; ++i) {
        printf("w[%zu]=%.4f ", i, n->ws[i]->data);
    }
    printf("b=%.4f", n->b->data);
}

Layer *layer_alloc(Arena *a, size_t n_in, size_t n_out) {
    Layer *layer = arena_alloc(a, sizeof(Layer)); 
    
    layer->neurons = arena_alloc(a, sizeof(Neuron*) * n_out);

    for (size_t i = 0; i < n_out; ++i) {
        layer->neurons[i] = neuron_alloc(a, n_in);
    }

    layer->n_in = n_in;
    layer->n_out = n_out;

    return layer;
}

void layer_print(Layer *l) {
    printf("Layer in=%zu out=%zu\n", l->n_in, l->n_out);
    for (size_t i = 0; i < l->n_out; ++i) {
        printf("\t\t");
        neuron_print(l->neurons[i]);
        printf("\n");
    }
}

MLP *mlp_alloc(Arena *a, size_t *dims, size_t dim_size) {
    MLP *mlp = arena_alloc(a, sizeof(MLP));
    
    mlp->layers = arena_alloc(a, sizeof(Layer*) * (dim_size - 1));
    mlp->dims = arena_alloc(a, sizeof(size_t) * dim_size);

    memcpy(mlp->dims, dims, sizeof(size_t) * dim_size);

    for (size_t i = 0; i < dim_size - 1; ++i) {
        mlp->layers[i] = layer_alloc(a, dims[i], dims[i + 1]);
    }
    
    mlp->dim_size = dim_size;

    return mlp;
}

void mlp_print(MLP *m) {
    printf("MLP ");
    for (size_t i = 0; i < m->dim_size; ++i) {
        if (i == 0) {
            printf("in=%zu ", m->dims[i]);
        } else if (i == m->dim_size - 1) {
            printf("out=%zu ", m->dims[i]);
        } else {
            printf("h[%zu]=%zu ", i, m->dims[i]);
        }
    }
    printf("\n");
    for (size_t i = 0; i < m->dim_size - 1; ++i) {
        printf("\t");
        layer_print(m->layers[i]);
    }
}

Value *neuron_forward(Arena *a, Neuron *n, Value **x, size_t x_size) {
    if (n->n_in != x_size) {
        printf("Invalid dimension");
        exit(1);
    }

    for (size_t i = 0; i < x_size; ++i) {
        x[i]->value_kind = VALUE_INPUT;
        sprintf(x[i]->label, "x[%zu]", i);
    }

    Value *out = value_create(a, 0);
    out->value_kind = VALUE_BOOTSTRAP;

    for (size_t i = 0; i < x_size; ++i) {
        // Value *w = value_create(a, n->ws[i]->data);
        // w->value_kind = VALUE_PARAM;
        // sprintf(w->label, "w[%zu]", i);
        Value *w = n->ws[i];
        w->value_kind = VALUE_PARAM;
        sprintf(w->label, "w[%zu]", i);
        
        Value *mul = value_mul(a, w, x[i]);
        out = value_add(a, out, mul);
    }

    // Value *b = value_create(a, n->b->data);
    Value *b = n->b;
    sprintf(b->label, "b");
    b->value_kind = VALUE_PARAM;
    out = value_add(a, out, b);
    out = value_tanh(a, out);

    return out;
}


Value **layer_forward(Arena *a, Layer *l, Value **x, size_t x_size) {
    if (l->n_in != x_size) {
        printf("Invalid dimension\n");
        exit(1);
    }
    
    Value **out = arena_alloc(a, sizeof(Value*) * l->n_out);

    for (size_t i = 0; i < l->n_out; ++i) {
        out[i] = neuron_forward(a, l->neurons[i], x, x_size);
    }

    return out;
}

/* Build the DAG */
Value **mlp_forward(Arena *a, MLP *m, Value **x, size_t x_size) {
    // I think that this will use a different arena ? 
    if (m->dims[0] != x_size) {
        printf("Invalid dimension\n");
        exit(1);
    }

    for (size_t i = 0; i < m->dim_size - 1; ++i) { // Layers
        x_size = m->dims[i];
        x = layer_forward(a, m->layers[i], x, x_size);
    }

    return x;
}

void neuron_zero_grad(Neuron *n) {
    for (size_t i = 0; i < n->n_in; ++i) {
        n->ws[i]->grad = 0;
    }
    n->b->grad = 0;
}

void layer_zero_grad(Layer *l) {
    for (size_t i = 0; i < l->n_out; ++i) {
        neuron_zero_grad(l->neurons[i]);
    }
}

void mlp_zero_grad(MLP *m) {
    for (size_t i = 0; i < m->dim_size - 1; ++i) {
        layer_zero_grad(m->layers[i]);
    }
}

void neuron_update(Neuron *n, double lr) {
    for (size_t i = 0; i < n->n_in; ++i) {
        n->ws[i]->data -= lr * n->ws[i]->grad;
    }
    n->b->data -= lr * n->b->grad;
}

void layer_update(Layer *l, double lr) {
    for (size_t i = 0; i < l->n_out; ++i) {
        neuron_update(l->neurons[i], lr);
    }
}

void mlp_update(MLP *m, double lr) {
    for (size_t i = 0; i < m->dim_size - 1; ++i) {
        layer_update(m->layers[i], lr);
    }   
}

Value *mse(Arena *a, Value *pred, Value *target) {
    Value *diff = value_sub(a, pred, target);

    Value *two = value_create(a, 2.0);
    Value *sq = value_pow(a, diff, two);

    return sq;
}