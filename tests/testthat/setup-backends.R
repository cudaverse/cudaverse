# The main package contract suite validates the stable CPU/torch behavior.
# Native hardware parity runs only when CUDAVERSE_NATIVE_TESTS=true.
options(cudaverse.cuda_backends = "torch")
