# B300 (Blackwell Ultra, sm_103) layer over the fork's stock CUDA 13.0 image.
#
# The stock image already carries a Blackwell-capable stack - torch 2.13 cu130, sglang-kernel 0.4.7,
# flashinfer 0.6.18, sgl-deep-gemm 0.2.0, nvidia-cutlass-dsl 4.6.2 - so this layer only pins what a JIT
# compile targets, ships FlashInfer's prebuilt jit-cache, and fails the build if any of it cannot see sm_103.
#
# Build with docker/enroot/build.sh --layer docker/b300, which builds the stock image first and passes it as BASE_IMAGE.
ARG BASE_IMAGE=sglang-fork:base
FROM ${BASE_IMAGE}

# Every JIT path compiles for B200 and B300 rather than probing the build host, which has no GPU
# => torch cpp extensions, flashinfer and DeepGEMM pick sm_100a/sm_103a
ENV TORCH_CUDA_ARCH_LIST="10.0a;10.3a" \
    FLASHINFER_CUDA_ARCH_LIST="10.0a 10.3a" \
    SGLANG_B300_IMAGE=1

# uv for installs into the image's venv - the stock runtime stage leaves its binary behind
RUN python3 -m pip install --no-cache-dir uv==0.9.5 && uv --version

COPY docker/b300/check_image.py /opt/sglang/check_image.py

# Import and arch checks that need no GPU
RUN python3 /opt/sglang/check_image.py --no-gpu
