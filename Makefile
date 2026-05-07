CC := g++
# Force NIX_ENFORCE_NO_NATIVE to be empty to allow -march=native
NIX_ENFORCE_NO_NATIVE :=
export NIX_ENFORCE_NO_NATIVE

# Flags
CFLAGS_COMMON := -std=c++17 -Wall -Wextra -Wpedantic -fopenmp -Iinclude -MMD -MP -march=native
RELEASE_FLAGS := -O3 -flto -DNDEBUG
DEBUG_FLAGS   := -g -Og -fsanitize=address -fsanitize=undefined

# Build Directories
BUILD_DIR := build
DEBUG_DIR := $(BUILD_DIR)/debug
RELEASE_DIR := $(BUILD_DIR)/release

# Default mode
MODE ?= release
ifeq ($(MODE),debug)
    CUR_BUILD_DIR := $(DEBUG_DIR)
    CFLAGS := $(CFLAGS_COMMON) $(DEBUG_FLAGS)
    LDFLAGS := -fsanitize=address -fsanitize=undefined
else
    CUR_BUILD_DIR := $(RELEASE_DIR)
    CFLAGS := $(CFLAGS_COMMON) $(RELEASE_FLAGS)
    LDFLAGS := -flto
endif

LDLIBS := -lm -fopenmp

MAKEFLAGS += -j$(shell nproc)

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LDLIBS := $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

TARGET := mnist_cnn
SERVER_TARGET := mnist_server
AUGMENT_TARGET := mnist_augment

COMMON_SRC := src/core/cnn.cpp \
               src/utils/tensor.cpp \
               src/layers/dense_layer.cpp \
               src/mnist/mnist_dataset.cpp \
               src/layers/pooling_layer.cpp \
               src/layers/conv_layer.cpp \
               src/layers/batchnorm_layer.cpp \
               src/loss/cross_entropy_loss.cpp \
               src/loss/mse_loss.cpp \
               src/layers/activation/activation.cpp \
               src/layers/activation/softmax_layer.cpp \
               src/layers/activation/relu_layer.cpp \
               src/layers/activation/sigmoid_layer.cpp \
               src/optimizers/sgd_optimizer.cpp \
               src/optimizers/adam_optimizer.cpp \
               src/layers/flatten_layer.cpp \
               src/layers/dropout_layer.cpp
CLI_SRC := src/core/main.cpp $(COMMON_SRC)
SERVER_SRC := src/core/server.cpp $(COMMON_SRC)
AUGMENT_SRC := src/mnist/data_augmentation.cpp

CLI_OBJ := $(CLI_SRC:%.cpp=$(CUR_BUILD_DIR)/%.o)
SERVER_OBJ := $(SERVER_SRC:%.cpp=$(CUR_BUILD_DIR)/%.o)
AUGMENT_OBJ := $(AUGMENT_SRC:%.cpp=$(CUR_BUILD_DIR)/%.o)

DEP := $(CLI_OBJ:.o=.d) $(SERVER_OBJ:.o=.d) $(AUGMENT_OBJ:.o=.d)

.PHONY: all clean debug fmt

all: $(TARGET) $(SERVER_TARGET) $(AUGMENT_TARGET)

debug:
	$(MAKE) MODE=debug

fmt:
	find src include -name "*.cpp" -o -name "*.hpp" | xargs -I{} clang-format -i -style=file {}

$(TARGET): $(CLI_OBJ) $(FORCE_REBUILD)
	$(CC) $(CFLAGS) -o $@ $(CLI_OBJ) $(LDFLAGS) $(LDLIBS)
	@mkdir -p $(BUILD_DIR)
	@echo $(MODE) > $(BUILD_DIR)/.last_mode

$(SERVER_TARGET): $(SERVER_OBJ) $(FORCE_REBUILD)
	$(CC) $(CFLAGS) -o $@ $(SERVER_OBJ) $(LDFLAGS) $(LDLIBS)
	@mkdir -p $(BUILD_DIR)
	@echo $(MODE) > $(BUILD_DIR)/.last_mode

$(AUGMENT_TARGET): $(AUGMENT_OBJ) $(FORCE_REBUILD)
	$(CC) $(CFLAGS) -o $@ $(AUGMENT_OBJ) $(LDFLAGS) $(LDLIBS) $(OPENCV_LDLIBS)
	@mkdir -p $(BUILD_DIR)
	@echo $(MODE) > $(BUILD_DIR)/.last_mode

$(CUR_BUILD_DIR)/src/mnist/data_augmentation.o: src/mnist/data_augmentation.cpp
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(CUR_BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) -r $(BUILD_DIR) $(TARGET) $(SERVER_TARGET) $(AUGMENT_TARGET)

.FORCE_REBUILD:

-include $(DEP)
