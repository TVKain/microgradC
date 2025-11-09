# Makefile for MicrogradC

CC      := gcc
CFLAGS  := -Wall -Wextra -O2 -std=c99 -Iinclude
LDFLAGS := -lm
AR      := ar
ARFLAGS := rcs
BUILD   := build
SRC_DIR := src
INC_DIR := include
EXAMPLES_DIR := examples

LIB_NAME := libmicrogradc.a

SRC_FILES := $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.c,$(BUILD)/%.o,$(SRC_FILES))

EXAMPLES := $(patsubst $(EXAMPLES_DIR)/%.c,$(BUILD)/%,$(wildcard $(EXAMPLES_DIR)/*.c))

.PHONY: all clean run examples

all: $(BUILD)/$(LIB_NAME) examples

$(BUILD):
	mkdir -p $(BUILD)

# Build static library
$(BUILD)/$(LIB_NAME): $(OBJ_FILES)
	$(AR) $(ARFLAGS) $@ $^

# Compile C sources
$(BUILD)/%.o: $(SRC_DIR)/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

# Build examples
examples: $(EXAMPLES)

$(BUILD)/%: $(EXAMPLES_DIR)/%.c $(BUILD)/$(LIB_NAME)
	$(CC) $(CFLAGS) $< -L$(BUILD) -lmicrogradc $(LDFLAGS) -o $@

# Run example
run: examples
	$(BUILD)/xor

clean:
	rm -rf $(BUILD)
