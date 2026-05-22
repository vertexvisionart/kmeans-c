# K-means demo build
#
# Targets:
#   make           build the demo binary
#   make run       build and execute (writes CSVs into data/)
#   make valgrind  run under valgrind with leak check
#   make clean     remove build artifacts

CC      = gcc
CFLAGS  = -std=c99 -Wall -Wextra -Wpedantic -O2 -I include
LDFLAGS = -lm

SRC_DIR = src
OBJ_DIR = build
BIN     = kmeans_demo

SRCS = $(SRC_DIR)/kmeans.c $(SRC_DIR)/main.c
OBJS = $(OBJ_DIR)/kmeans.o $(OBJ_DIR)/main.o

.PHONY: all run clean valgrind

all: $(OBJ_DIR) $(BIN)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/kmeans.o: $(SRC_DIR)/kmeans.c include/kmeans.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/main.o: $(SRC_DIR)/main.c include/kmeans.h
	$(CC) $(CFLAGS) -c $< -o $@

run: all
	mkdir -p data
	./$(BIN)

valgrind: all
	mkdir -p data
	valgrind --leak-check=full --error-exitcode=1 ./$(BIN)

clean:
	rm -rf $(OBJ_DIR) $(BIN)
