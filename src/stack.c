#include "stack.h"
#include <stdlib.h>
#include <string.h>

Stack *stack_create(void) {
    Stack *s = malloc(sizeof(*s));
    if (!s) return NULL;
    s->capacity = 16;
    s->size = 0;
    s->items = malloc(sizeof(void *) * s->capacity);
    if (!s->items) {
        free(s);
        return NULL;
    }
    return s;
}

void stack_destroy(Stack *s) {
    if (!s) return;
    free(s->items);
    free(s);
}

bool stack_empty(const Stack *s) {
    return s->size == 0;
}

bool stack_contains(const Stack *s, void *item) {
    for (size_t i = 0; i < s->size; ++i) {
        if (s->items[i] == item) {
            return true;
        }
    }
    return false;
}

void stack_push(Stack *s, void *item) {
    if (!s) return;
    if (s->size == s->capacity) {
        s->capacity *= 2;
        void **tmp = realloc(s->items, sizeof(void *) * s->capacity);
        if (!tmp) return; /* allocation failed, silently fail */
        s->items = tmp;
    }
    s->items[s->size++] = item;
}

void *stack_pop(Stack *s) {
    if (!s || s->size == 0) return NULL;
    return s->items[--s->size];
}

void *stack_peek(const Stack *s) {
    if (!s || s->size == 0) return NULL;
    return s->items[s->size - 1];
}
