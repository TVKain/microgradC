#ifndef STACK_H
#define STACK_H

#include <stddef.h>
#include <stdbool.h>

/* Opaque stack type for pointers */

typedef struct Stack Stack;

struct Stack {
    void **items;
    size_t size;
    size_t capacity;
};

/* Create a new stack */
Stack *stack_create(void);

/* Destroy stack and free memory */
void stack_destroy(Stack *s);

/* Push a pointer onto the stack */
void stack_push(Stack *s, void *item);

/* Pop a pointer from the stack. Returns NULL if empty */
void *stack_pop(Stack *s);

/* Peek at the top of the stack without removing. Returns NULL if empty */
void *stack_peek(const Stack *s);

bool stack_contains(const Stack *s, void *item);

/* Check if stack is empty */
bool stack_empty(const Stack *s);

#endif /* STACK_H */
