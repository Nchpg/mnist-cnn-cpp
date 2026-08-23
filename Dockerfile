# ---- Build stage ----
FROM debian:bookworm-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ make libgomp1 nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only what is needed to build the server (no OpenCV / augment target)
COPY Makefile ./
COPY include ./include
COPY src ./src

# Portable baseline, not the Makefile's native: the CI runner is not the host
# that runs the image.
ARG ARCH=x86-64-v2

RUN make mnist_server ARCH=${ARCH}

# ---- Runtime stage ----
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Which pretrained model to bake into the image (override with --build-arg).
ARG MODEL_FILE=mnist_cnn.model

COPY --from=builder /app/mnist_server ./mnist_server
COPY public ./public
COPY ${MODEL_FILE} ./model.model

EXPOSE 8080

CMD ["./mnist_server", "model.model", "8080"]
