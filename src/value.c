
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

static void backward_neg(Value *v) {
    v->prev[0]->grad += -v->grad;
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
 * y = e^x
 * dy/dx = e^x = y
 */
static void backward_exp(Value *v) {
    v->prev[0]->grad += v->data * v->grad;
}


/** 
 * y = log(x)
 * dy/dx = 1/x 
 */
static void backward_log(Value *v) {
    Value *a = v->prev[0];
    a->grad += (1 / a->data) * v->grad;
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

/**
 * dy/dx = y(1 - y)
 */
static void backward_sigmoid(Value *v) {
    Value *prev = v->prev[0];
    prev->grad += (v->data) * (1 - v->data) * v->grad;
}

static void backward_relu(Value *v) {
    Value *prev = v->prev[0];

    if (prev->data > 0) {
        prev->grad += v->grad;
    } 
}

Value *value_alloc(Arena *a, double data) {
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
    Value *out = value_alloc(a,  v1->data + v2->data);
    out->backward = backward_add;
    out->n_prev = 2;
    out->prev = arena_alloc(a, sizeof(Value*) * 2);
    out->prev[0] = v1;
    out->prev[1] = v2;
    out->grad = 0.0;
    out->op = OP_ADD;
    
    return out;
}

Value *value_neg(Arena *a, Value *v1) {
    Value *out = value_alloc(a,  -v1->data);
    out->backward = backward_neg;
    out->n_prev = 1;
    out->prev = arena_alloc(a, sizeof(Value*));
    out->prev[0] = v1;
    out->grad = 0.0;
    out->op = OP_NEG;
    
    return out;
}

Value *value_sub(Arena *a, Value *v1, Value *v2) {
    Value *out = value_alloc(a, v1->data - v2->data);
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
    Value *out = value_alloc(a, v1->data * v2->data);
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
    Value *out = value_alloc(a, pow(v1->data, v2->data));

    out->backward = backward_pow;
    out->n_prev = 2;
    out->prev = arena_alloc(a, sizeof(Value*) * 2);
    out->prev[0] = v1;
    out->prev[1] = v2;
    out->grad = 0.0;
    out->op = OP_POW;

    return out;
}

Value *value_exp(Arena *a, Value *v1) {
    Value *out = value_alloc(a, exp(v1->data));

    out->backward = backward_exp;
    out->n_prev = 1;
    out->prev = arena_alloc(a, sizeof(Value*));
    out->prev[0] = v1;
    out->grad = 0.0;
    out->op = OP_EXP;

    return out;
}

Value *value_log(Arena *a, Value *v1) {
    if (v1->data <= 0) {
        printf("Invalid data for log\n");
        exit(1);
    }

    Value *out = value_alloc(a, log(v1->data));

    out->backward = backward_log;
    out->n_prev = 1;
    out->prev = arena_alloc(a, sizeof(Value*));
    out->prev[0] = v1;
    out->grad = 0.0;
    out->op = OP_LOG;

    return out;
}


Value *value_div(Arena *a, Value *v1, Value *v2) {
    if (v2->data == 0) {
        printf("Div by zero");
        exit(1);
    }

    Value *out = value_alloc(a, v1->data / v2->data);

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
    Value *out = value_alloc(a, tanh(v1->data));
    out->grad = 0.0;
    out->n_prev = 1;
    out->prev = arena_alloc(a, sizeof(Value*));
    out->prev[0] = v1;
    out->backward = backward_tanh;
    out->op = OP_TANH;
    return out;
}

Value *value_sigmoid(Arena *a, Value *v1) {
    Value *out = value_alloc(a, 1 / (1 + exp(-v1->data)));
    out->grad = 0.0;
    out->n_prev = 1;
    out->prev = arena_alloc(a, sizeof(Value*));
    out->prev[0] = v1;
    out->backward = backward_sigmoid;
    out->op = OP_SIGMOID;
    return out;
}

Value *value_relu(Arena *a, Value *v1) {
    double data = v1->data;

    if (data < 0) {
        data = 0;
    }

    Value *out = value_alloc(a, data);
    out->grad = 0.0;
    out->n_prev = 1;
    out->prev = arena_alloc(a, sizeof(Value*));
    out->prev[0] = v1;
    out->backward = backward_relu;
    out->op = OP_RELU;
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

    // arena_reset(a);
    stack_destroy(visited);
    stack_destroy(s);
}


Value *mse(Arena *a, Value **pred, Value **target, size_t size) {
    Value *out = value_alloc(a, 0);

    Value *two = value_alloc(a, 2.0);

    for (size_t i = 0; i < size; ++i) {
        Value *sub = value_sub(a, pred[i], target[i]);
        Value *pow = value_pow(a, sub, two);
        out = value_add(a, out, pow);
    }

    Value *n = value_alloc(a, (double)size);

    out = value_div(a, out, n);
    return out;
}

/**
 * Cross-entropy loss for a single example
 * pred: array of Value* representing softmax probabilities
 * target: index value of correct prediction 
 * size: number of classes
 *
 * softmax(x) = softmax(x - a)
 * 
 * L = - log(pred[target])
 */
Value *cross_entropy(Arena *a, Value **preds, Value *target, size_t size) {
    size_t target_index = (size_t) target->data;

    if (target_index >= size) {
        printf("Invalid target index\n");
        exit(1);
    }

    // Max logit stability trick
    Value *max_logit = preds[0];
    for (size_t i = 1; i < size; i++) {
        if (preds[i]->data > max_logit->data) {
            max_logit = preds[i];
        }
    }

    Value *sum_exp = value_alloc(a, 0);
    Value *target_pred = NULL;

    for (size_t i = 0; i < size; ++i) {
        Value *sub = value_sub(a, preds[i], max_logit);
        Value *exp_pred = value_exp(a, sub);

        if (target_index == i) {
            target_pred = exp_pred;
        }

        sum_exp = value_add(a, sum_exp, exp_pred);
    }

    Value *prob_target = value_div(a, target_pred, sum_exp);
    Value *log_prob = value_log(a, prob_target);
    Value *loss = value_neg(a, log_prob);

    return loss;
}

Value **soft_max(Arena *a, Value **logits, size_t size) {
    // Find max logit for numerical stability
    Value *max_logit = logits[0];
    for (size_t i = 1; i < size; i++) {
        if (logits[i]->data > max_logit->data) {
            max_logit = logits[i];
        }
    }

    Value **exp_vals = arena_alloc(a, sizeof(Value*) * size);
    Value *sum_exp = value_alloc(a, 0.0);

    // Compute exponentials
    for (size_t i = 0; i < size; ++i) {
        Value *sub = value_sub(a, logits[i], max_logit);
        exp_vals[i] = value_exp(a, sub);
        sum_exp = value_add(a, sum_exp, exp_vals[i]);
    }

    // Compute softmax output
    Value **out = arena_alloc(a, sizeof(Value*) * size);
    for (size_t i = 0; i < size; ++i) {
        out[i] = value_div(a, exp_vals[i], sum_exp);
    }

    return out;
}

const char* op_to_string(Op_Kind op) {
    switch (op) {
        case OP_NONE:  return "NONE";
        case OP_ADD:   return "ADD";
        case OP_SUB:   return "SUB";
        case OP_MUL:   return "MUL";
        case OP_DIV:   return "DIV";
        case OP_TANH:  return "TANH";
        case OP_POW:   return "POW";
        case OP_EXP:   return "EXP";
        case OP_LOG:   return "LOG";
        case OP_NEG:   return "NEG";
        case OP_SIGMOID: return "SIGMOID";
        case OP_RELU:  return "RELU";
        default:       return "UNKNOWN";
    }
}

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

const char* value_kind_to_color(Value_Kind value_kind) {
    switch (value_kind) {
        case VALUE_INPUT:     return "gold";
        case VALUE_PARAM:     return "lightcyan";
        case VALUE_BOOTSTRAP: return "lightpink";
        default: return "white";
    }
}

// ------------------------ Visited tracking ------------------------

int already_visited(Value *v, Value **visited, size_t n_visited) {
    for (size_t i = 0; i < n_visited; i++) {
        if (visited[i] == v) return 1;
    }
    return 0;
}

// ------------------------ Recursive DOT printer ------------------------

static inline unsigned int hash_ptr(void *ptr) {
    uintptr_t p = (uintptr_t)ptr;
    // XOR upper and lower bits, then modulo some number if needed
    return (unsigned int)(((p >> 4) ^ (p & 0xFFFF)) % 1000);
}
void print_dot(Value *v, FILE *f, Value **visited, size_t *n_visited, Value *root) {
    if (!v) return;

    if (already_visited(v, visited, *n_visited)) {
        printf("[debug] Already visited node %u\n", hash_ptr(v));
        return;
    }

    visited[*n_visited] = v;
    (*n_visited)++;

    unsigned int id = hash_ptr(v);  // hashed ID for readability

    // Determine color
    const char *color;
    if (v == root) {
        color = "red";
    } else if (v->op != OP_NONE) {
        color = op_to_color(v->op);
    } else {
        color = value_kind_to_color(v->value_kind);
    }

    // Debug print
    printf("[debug] Node %u: data=%.4f grad=%.4f op=%s label='%s' n_prev=%zu\n",
           id, v->data, v->grad, op_to_string(v->op), v->label, v->n_prev);

    // Node label in DOT file
    if (v->label[0] != '\0') {
        fprintf(f, "  %u [label=\"%s\\ndata=%.4f\\ngrad=%.4f\\nid=%u\\nop=%s\", style=filled, fillcolor=%s];\n",
                id, v->label, v->data, v->grad, id, op_to_string(v->op), color);
    } else {
        fprintf(f, "  %u [label=\"data=%.4f\\ngrad=%.4f\\nid=%u\\nop=%s\", style=filled, fillcolor=%s];\n",
                id, v->data, v->grad, id, op_to_string(v->op), color);
    }

    // Edges
    for (size_t i = 0; i < v->n_prev; i++) {
        fprintf(f, "  %u -> %u;\n", hash_ptr(v->prev[i]), id);
        printf("[debug] Edge: %u -> %u\n", hash_ptr(v->prev[i]), id);
        print_dot(v->prev[i], f, visited, n_visited, root);
    }
}



// ------------------------ Public function to export DAG ------------------------
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

    Value *visited[10000];
    size_t n_visited = 0;

    print_dot(root, f, visited, &n_visited, root);  // <-- pass root

    fprintf(f, "}\n");
    fclose(f);

    char cmd[512];
    snprintf(cmd, sizeof(cmd), "dot -Tpng %s -o %s.png", dotfile, filename);
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "Graphviz command failed\n");
    } else {
        printf("DAG exported to %s.png\n", filename);
    }
}

