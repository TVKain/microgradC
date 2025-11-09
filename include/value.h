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
    OP_POW 
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

Value *value_create(Arena *a, double data);
Value *value_add(Arena *a, Value *v1, Value *v2);
Value *value_sub(Arena *a, Value *v1, Value *v2);
Value *value_mul(Arena *a, Value *v1, Value *v2);
Value *value_div(Arena *a, Value *v1, Value *v2);
Value *value_pow(Arena *a, Value *v1, Value *v2);
Value *value_tanh(Arena *a, Value *v1);

void value_backward(Arena *a, Value *v);

// void print_dag(Value *root);
void export_dag_png(Value *root, const char *filename);

#endif