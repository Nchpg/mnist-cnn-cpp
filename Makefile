CC := g++
# Force NIX_ENFORCE_NO_NATIVE to be empty to allow -march=native
NIX_ENFORCE_NO_NATIVE :=
export NIX_ENFORCE_NO_NATIVE
CFLAGS_COMMON := -std=c++17 -Wall -Wextra -Wpedantic -fopenmp -Iinclude -MMD -MP -O3 -march=native
CFLAGS_DEBUG := $(CFLAGS_COMMON) -g -Og -fsanitize=address -fsanitize=undefined
CFLAGS := $(CFLAGS_COMMON)
LDLIBS := -lm -fopenmp

MAKEFLAGS += -j$(shell nproc)

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LDLIBS := $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

TARGET := mnist_cnn
SERVER_TARGET := mnist_server
AUGMENT_TARGET := mnist_augment

COMMON_SRC := src/core/cnn.cpp \
             src/layers/dense_layer.cpp \
             src/data/mnist_dataset.cpp \
             src/utils/matrix.cpp \
             src/utils/tensor_batch.cpp \
             src/layers/pooling_layer.cpp \
             src/layers/conv_layer.cpp \
             src/layers/batchnorm_layer.cpp \
             src/loss/cross_entropy_loss.cpp \
             src/layers/activation/activation.cpp \
             src/optimizers/sgd_optimizer.cpp \
             src/optimizers/adam_optimizer.cpp \
             src/layers/flatten_layer.cpp \
             src/layers/dropout_layer.cpp \
             src/layers/relu_layer.cpp
CLI_SRC := src/core/main.cpp $(COMMON_SRC)
SERVER_SRC := src/core/server.cpp $(COMMON_SRC)
AUGMENT_SRC := src/data/data_augmentation.cpp

CLI_OBJ := $(CLI_SRC:.cpp=.o)
SERVER_OBJ := $(SERVER_SRC:.cpp=.o)
AUGMENT_OBJ := $(AUGMENT_SRC:.cpp=.o)

DEP := $(CLI_OBJ:.o=.d) $(SERVER_OBJ:.o=.d) $(AUGMENT_OBJ:.o=.d)

.PHONY: all clean debug fmt

all: $(TARGET) $(SERVER_TARGET) $(AUGMENT_TARGET)

debug: CFLAGS = $(CFLAGS_DEBUG)
debug: clean all

fmt:
	find src include -name "*.cpp" -o -name "*.hpp" | xargs -I{} clang-format -i -style=file {}

$(TARGET): $(CLI_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(SERVER_TARGET): $(SERVER_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(AUGMENT_TARGET): $(AUGMENT_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) $(OPENCV_LDLIBS)

src/data/data_augmentation.o: src/data/data_augmentation.cpp
	$(CC) $(CFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(CLI_OBJ) $(SERVER_OBJ) $(AUGMENT_OBJ) $(DEP) $(TARGET) $(SERVER_TARGET) $(AUGMENT_TARGET)

-include $(DEP)