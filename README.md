# Mango

C++20 tensor library with reverse-mode autodiff. Tensors support strided layouts, scalar broadcasting, and a runtime operation graph for backpropagation. Matmul uses BLAS (Apple Accelerate on macOS).

## Build & run

**Requirements:** CMake ≥ 3.16, C++20 compiler. Non-Apple platforms need a BLAS library (`find_package(BLAS)`).

### Demo (`main.cpp`)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build -j
./build/mango
```

### Tests

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
# or: ./build/mango_tests
```

### Using `mango_lib` in your project

Link the static library and include headers from `include/`:

```cmake
add_subdirectory(path/to/Mango)   # or import mango_lib however you prefer
target_link_libraries(your_target PRIVATE mango_lib)
```

```cpp
#include "tensor_lib_headers/tensor.h"

using mango::Tensor;
using mango::Shape;
using mango::DType;
```

Public headers live under `include/tensor_lib_headers/` and `include/auto_grad_node_headers/`.

---

## Tensor

A `Tensor` is a typed, strided view over shared `Storage`, plus optional autograd state (`grad_fn_`, `grad_tensor_`).

### Construction

```cpp
Tensor a(Shape{2, 3}, DType::F32);                    // empty buffer
Tensor b({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});          // from values
Tensor w = Tensor::randn<double>(Shape{3, 4}, true);  // learnable (tracks grad)
Tensor z = Tensor::zeros(Shape{2, 2}, DType::F64);
Tensor o = Tensor::ones<float>(Shape{2, 2});
```

Supported dtypes: `F32`, `F64`, `I32`, `I64`, `B`.  
`learnable=true` allocates a zero gradient buffer and marks the tensor as a parameter leaf.

Layout helpers: `view`, `reshape` (contiguous only), `clone`, `transpose` / `transpose_inplace`, `contiguous`, `to(DType)`.

### Broadcasting

Elementwise ops support **scalar-like** broadcasting only (`numel() == 1`, any rank including `Shape{}`):

```cpp
Tensor x({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
Tensor s({2.0}, Shape{});
Tensor y = x + s;   // adds 2 to every element
```

Same-shape tensors also work. General NumPy-style broadcasting is not supported. In backward, a scalar parameter that was broadcast in forward receives a **summed** upstream gradient via `accum_grad`.

### Recording vs non-recording

| Kind          | API                                                                                                       | Autograd                                                               |
| ------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Non-recording | `add_nr`, `sub_nr`, `mult_nr`, `matmul_nr`, `square_nr`, `relu_nr`, `negate_nr`, `sum_nr`, `transpose_nr` | No `grad_fn`; use inside backward kernels or when grads are not needed |
| Recording     | `+`, `-`, `*`, `matmul`, `square`, `relu`, `sum`, `mean`, `transpose`                                     | Attaches a `*Backward` node if any input `requires_grad()`             |

In-place `+=`, `-=`, `*=` update data only (no graph). Reductions `max` / `min` have no autograd yet.

```cpp
Tensor out = a.matmul(b);     // records MatMulBackward if needed
Tensor raw = a.matmul_nr(b);  // same math, no graph
```

---

## Autograd

### How it works

1. **Forward:** Recording ops create a DAG of `Node` subclasses (`AddBackward`, `MulBackward`, …). Each result stores `grad_fn` pointing at the node that produced it.
2. **Backward:** `loss.backward()` seeds `∂L/∂loss = 1` and calls `grad_fn->backwardPass`. Each node applies its local Jacobian, then `propagate`s into parents.
3. **Leaves:** Learnable tensors have no `grad_fn`; gradients accumulate in `grad_tensor_` via `accum_grad`.

### `Node` structure

- parents\* // input tensors for this op
- consumer_count\* // how many downstream ops use this node's output
- received*gradients* // how many of those have contributed in backward
- add*parent() // register parent; bump parent's consumer_count*
- propagate() // accum_grad + fire parent only when all consumers done
- backwardPass(grad_out) // op-specific local gradient

**Consumer counting (DAG-safe):** If the same intermediate is used twice, its node waits until both paths deliver gradients before running `backwardPass`. Leaves just accumulate. Rebuild the graph each training step (one `backward()` per forward); counters are not reset for a second backward on the same graph.

### Example graph

```cpp
Tensor x({1.0, 2.0}, Shape{2}, /*learnable=*/true);
Tensor w({3.0, 4.0}, Shape{2}, true);

Tensor t1 = x * w;           // MulBackward(x, w)
Tensor t2 = t1 + x;          // AddBackward(t1, x)  — x used again
Tensor loss = t2.square().mean();

loss.backward();
// x.grad() ≈ ∂/∂x (mean((x*w + x)^2))
// w.grad() ≈ ∂/∂w of the same
```

Forward DAG:

```text
x ──┬──(*)── t1 ──(+)── t2 ──(square)── t3 ──(mean)── loss
w ──┘         ▲
              └── x (second use)
```

`x` is a leaf: both paths call `accum_grad` into the same `grad_tensor_`. Shared intermediates with a `grad_fn` use `consumer_count_` so upstream backward runs once with the full gradient.

Training step pattern:

```cpp
loss.backward();
w -= w.grad()->mult_nr(Tensor({lr}, Shape{}));
x.zero_grad();
w.zero_grad();
```
