
#include "value.h"
#include "arena.h"
#include "stack.h"

#include <math.h>
#include <stdlib.h>

static void backward_add(Value *v) {
    v->prev[0]->grad += v->grad;
    v->prev[1]->grad += v->grad;
}

static void backward_sub(Value *v) {
    v->prev[0]->grad += v->grad;
    v->prev[1]->grad += -v->grad;
}

static void backward_mul(Value *v) {
    v->prev[0]->grad += v->prev[1]->data * v->grad;
    v->prev[1]->grad += v->prev[0]->data * v->grad;
}


/**
 * y = a^b 
 * dy/da = b*a^(b-1)
 * 
 * y = e^(ln(a^b)) = e^(b*ln(a))
 * dy/db = e^(b*ln(a))*ln(a) = a^b*ln(a) = y*ln(a), a > 0 else NAN
 */
static void backward_pow(Value *v) {
    Value *a = v->prev[0];
    Value *b = v->prev[1];

    a->grad += (b->data) * pow(a->data, b->data - 1) * v->grad;

    if (a -> data > 0) {
        b->grad += v->data * log(a->data) * v->grad;
    } else {
        b->grad = NAN;
    }
}

/** 
 * y = a / b
 * dy/da = 1 / b
 * dy/db = -a / b^2
 * b != 0
 */
static void backward_div(Value *v) {
    Value *a = v->prev[0];
    Value *b = v->prev[1];

    a->grad += (1 / b->data) * v->grad;
    b->grad += (-a->data / (b->data * b->data)) * v->grad;
}

static void backward_tanh(Value *v) {
    v->prev[0]->grad += (1 - v->data * v->data) * v->grad;
}

Value *value_create(Arena *a, double data) {
    Value *v = arena_alloc(a, sizeof(Value));
    v->data = data;
    v->grad = 0.0; 
    v->prev = NULL;
    v->n_prev = 0;
    v->value_kind = VALUE_NONE;
    v->backward = NULL;
    v->op = OP_NONE;
    v->label[0] = '\0';
    return v;
}

Value *value_add(Arena *a, Value *v1, Value *v2) {
    Value *out = value_create(a,  v1->data + v2->data);
    out->backward = backward_add;
    out->n_prev = 2;
    out->prev = arena_alloc(a, sizeof(Value*) * 2);
    out->prev[0] = v1;
    out->prev[1] = v2;
    out->grad = 0.0;
    out->op = OP_ADD;
    
    return out;
}

Value *value_sub(Arena *a, Value *v1, Value *v2) {
    Value *out = value_create(a, v1->data - v2->data);
    out->backward = backward_sub;
    out->n_prev = 2;
    out->prev = arena_alloc(a, sizeof(Value*) * 2);
    out->prev[0] = v1;
    out->prev[1] = v2;
    out->grad = 0.0;
    out->op = OP_SUB;
    
    return out;
}

Value *value_mul(Arena *a, Value *v1, Value *v2) {
    Value *out = value_create(a, v1->data * v2->data);
    out->backward = backward_mul;
    out->n_prev = 2;
    out->prev = arena_alloc(a, sizeof(Value*) * 2);
    out->prev[0] = v1;
    out->prev[1] = v2;
    out->grad = 0.0;
    out->op = OP_MUL;
    
    return out;
}

Value *value_pow(Arena *a, Value *v1, Value *v2) {
    Value *out = value_create(a, pow(v1->data, v2->data));

    out->backward = backward_pow;
    out->n_prev = 2;
    out->prev = arena_alloc(a, sizeof(Value*) * 2);
    out->prev[0] = v1;
    out->prev[1] = v2;
    out->grad = 0.0;
    out->op = OP_POW;

    return out;
}

Value *value_div(Arena *a, Value *v1, Value *v2) {
    if (v2->data == 0) {
        printf("Div by zero");
        exit(1);
    }

    Value *out = value_create(a, v1->data / v2->data);

    out->op = OP_DIV;
    out->n_prev = 2;
    out->prev = arena_alloc(a, sizeof(Value*) * 2);
    out->prev[0] = v1;
    out->prev[1] = v2;
    out->grad = 0.0;
    out->backward = backward_div;
    
    return out;
}

Value *value_tanh(Arena *a, Value *v1) {
    Value *out = value_create(a, tanh(v1->data));
    out->grad = 0.0;
    out->n_prev = 1;
    out->prev = arena_alloc(a, sizeof(Value*));
    out->prev[0] = v1;
    out->backward = backward_tanh;
    out->op = OP_TANH;
    return out;
}

void value_backward(Arena *a, Value *v) {
    // Build topo
    Stack *s = stack_create();
    Stack *visited = stack_create();

    stack_push(s, v);

    while (!stack_empty(s)) {
        Value *node = stack_pop(s);

        if (stack_contains(visited, node)) {
            continue;
        }

        stack_push(visited, node);

        for (size_t i = 0; i < node->n_prev; ++i) {
            stack_push(s, node->prev[i]);
        }
    }

    v->grad = 1.0;

    for (size_t i = 0; i < visited->size; ++i) {
        Value *node = (Value*) visited->items[i];
        if (node->backward) {
            node->backward(node);
        }
    }

    arena_reset(a);
    stack_destroy(visited);
    stack_destroy(s);
}



// Helper to convert Op_Kind to string
const char* op_to_string(Op_Kind op) {
    switch (op) {
        case OP_NONE:  return "NONE";
        case OP_ADD:   return "ADD";
        case OP_SUB:   return "SUB";
        case OP_MUL:   return "MUL";
        case OP_DIV:   return "DIV";
        case OP_TANH:  return "TANH";
        case OP_POW:   return "POW";
        default:       return "UNKNOWN";
    }
}

// Helper to pick node color based on op
const char* op_to_color(Op_Kind op) {
    switch (op) {
        case OP_NONE: return "lightgray";
        case OP_ADD:  return "lightgreen";
        case OP_SUB:  return "orange";
        case OP_MUL:  return "lightblue";
        case OP_DIV:  return "pink";
        case OP_TANH: return "yellow";
        case OP_POW:  return "violet";
        default:      return "white";
    }
}

const char *value_kind_to_color(Value_Kind value_kind) {
    switch (value_kind) {
        case VALUE_INPUT: return "gold";        // leaf input nodes
        case VALUE_PARAM: return "lightcyan";  // parameters
        case VALUE_BOOTSTRAP: return "lightpink"; // other special nodes
        default: return "white";               // fallback
    }
}
static void print_dot(Value *v, FILE *f, int *visited) {
    if (!v) return;

    int id = (int)((uintptr_t)v % 10000); // unique-ish ID
    if (visited[id]) return;
    visited[id] = 1;

    const char *color = "blue";
    if (v->op != OP_NONE) {
        color = op_to_color(v->op);
    } else {
        color = value_kind_to_color(v->value_kind);
    }

    // Node label: data, grad, op, and optional user label
    if (v->label[0] != '\0') {
        fprintf(f, "  %d [label=\"%s\\ndata=%.4f\\ngrad=%.4f\\nop=%s\", style=filled, fillcolor=%s];\n",
                id, v->label, v->data, v->grad, op_to_string(v->op), color);
    } else {
       fprintf(f, "  %d [label=\"data=%.4f\\ngrad=%.4f\\nop=%s\", style=filled, fillcolor=%s];\n",
              id, v->data, v->grad, op_to_string(v->op), color);
    }

    // Edges to parents
    for (size_t i = 0; i < v->n_prev; i++) {
        int prev_id = (int)((uintptr_t)v->prev[i] % 10000);
        fprintf(f, "  %d -> %d;\n", prev_id, id);
        print_dot(v->prev[i], f, visited);
    }
}
// Public function to export DAG to PNG
void export_dag_png(Value *root, const char *filename) {
    char dotfile[256];
    snprintf(dotfile, sizeof(dotfile), "%s.dot", filename);

    FILE *f = fopen(dotfile, "w");
    if (!f) {
        perror("fopen");
        return;
    }

    fprintf(f, "digraph G {\n");
    fprintf(f, "  node [shape=box, fontname=\"Courier\"];\n");

    int visited[10000] = {0};
    print_dot(root, f, visited);

    fprintf(f, "}\n");
    fclose(f);

    // Generate PNG using Graphviz
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "dot -Tpng %s -o %s.png", dotfile, filename);
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "Graphviz command failed\n");
    } else {
        printf("DAG exported to %s.png\n", filename);
    }
}