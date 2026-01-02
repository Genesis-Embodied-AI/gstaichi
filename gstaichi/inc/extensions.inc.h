// Lists of extension features
PER_EXTENSION(sparse)       // Sparse data structures
PER_EXTENSION(quant)        // Quantization
PER_EXTENSION(mesh)         // MeshGsTaichi
PER_EXTENSION(quant_basic)  // Basic operations in quantization
PER_EXTENSION(data64)       // 64-bit data types (int64, uint64, float64)
PER_EXTENSION(int64)        // 64-bit integer types (int64, uint64) - subset of data64
PER_EXTENSION(adstack)    // For keeping the history of mutable local variables
PER_EXTENSION(bls)        // Block-local storage
PER_EXTENSION(assertion)  // Run-time asserts in GsTaichi kernels
PER_EXTENSION(extfunc)    // Invoke external functions or backend source
