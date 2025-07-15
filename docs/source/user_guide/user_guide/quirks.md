# Quirks

Things you may find surprising.

## Augmented assigns are atomic

The following are atomic:
- `+=`
- `-=`
- `*=`
- etc ...

Since writing these to mean atomic doesn't clearly express intent (did you write them intentionally meaning atomic? Or did you not realize they are atomic?), we recommend writing them out as atomics explicitly:

- `atomic_add`
- `atomic_sub`
- `atomic_mul`
- etc ...

In the future, we may explicitly deprecate augmented assigns.
