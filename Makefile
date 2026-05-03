CC := g++
CFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -fopenmp -Iinclude -MMD -MP -O3 -march=native
LDLIBS := -lm -fopenmp

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LDLIBS := $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

TARGET := mnist_cnn
SERVER_TARGET := mnist_server
AUGMENT_TARGET := mnist_augment

COMMON_SRC := src/cnn.cpp src/model_config.cpp src/dense_layer.cpp \
               src/mnist_dataset.cpp src/matrix.cpp \
               src/pooling_layer.cpp \
               src/conv_layer.cpp
CLI_SRC := src/main.cpp $(COMMON_SRC)
SERVER_SRC := src/server.cpp $(COMMON_SRC)
AUGMENT_SRC := src/data_augmentation.cpp

CLI_OBJ := $(CLI_SRC:.cpp=.o)
SERVER_OBJ := $(SERVER_SRC:.cpp=.o)
AUGMENT_OBJ := $(AUGMENT_SRC:.cpp=.o)

DEP := $(CLI_OBJ:.o=.d) $(SERVER_OBJ:.o=.d) $(AUGMENT_OBJ:.o=.d)

.PHONY: all clean

# Compile désormais les 3 exécutables
all: $(TARGET) $(SERVER_TARGET) $(AUGMENT_TARGET)

$(TARGET): $(CLI_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(SERVER_TARGET): $(SERVER_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(AUGMENT_TARGET): $(AUGMENT_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) $(OPENCV_LDLIBS)

src/data_augmentation.o: src/data_augmentation.cpp
	$(CC) $(CFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(CLI_OBJ) $(SERVER_OBJ) $(AUGMENT_OBJ) $(DEP) $(TARGET) $(SERVER_TARGET) $(AUGMENT_TARGET)

-include $(DEP)