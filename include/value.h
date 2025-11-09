#ifndef VALUE_H
#define VALUE_H

#include "arena.h"

#include <stddef.h>
#include <stdbool.h>


typedef enum Op_Kind {
    OP_NONE=0,
    OP_ADD,
    OP_SUB,
    OP_MUL, 
    OP_DIV,
    OP_TANH,
    OP_POW,
    OP_EXP,
    OP_LOG,
    OP_NEG,
    OP_SIGMOID,
    OP_RELU
} Op_Kind;

typedef struct Value Value;

/**
 * Just for visualization
 */
typedef enum Value_Kind {
    VALUE_PARAM,
    VALUE_INPUT,
    VALUE_BOOTSTRAP,
    VALUE_NONE
} Value_Kind;

struct Value {
    double data;
    double grad; 

    Value **prev;
    size_t n_prev;

    void (*backward)(Value *v);
    
    Op_Kind op;

    // Visualization only
    Value_Kind value_kind;
    char label[32];
};

Value *value_alloc(Arena *a, double data);
Value *value_add(Arena *a, Value *v1, Value *v2);
Value *value_sub(Arena *a, Value *v1, Value *v2);
Value *value_neg(Arena *a, Value *v1);
Value *value_mul(Arena *a, Value *v1, Value *v2);
Value *value_div(Arena *a, Value *v1, Value *v2);
Value *value_exp(Arena *a, Value *v1);
Value *value_log(Arena *a, Value *v1);
Value *value_pow(Arena *a, Value *v1, Value *v2);
Value *value_tanh(Arena *a, Value *v1);
Value *value_relu(Arena *a, Value *v1);
Value *value_sigmoid(Arena *a, Value *v1);

void value_backward(Arena *a, Value *v);

Value **soft_max(Arena *a, Value **logits, size_t size);
Value *mse(Arena *a, Value **pred, Value **target, size_t size);
Value *cross_entropy(Arena *a, Value **pred, Value *target, size_t size);

// void print_dag(Value *root);
void export_dag_png(Value *root, const char *filename);


#endif