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

Optional vDSP elementwise acceleration (Apple only; default ON on macOS):

```bash
cmake -S . -B build -DMANGO_ENABLE_VDSP=ON    # contiguous F32/F64 add/sub/mul/neg via vDSP
cmake -S . -B build -DMANGO_ENABLE_VDSP=OFF   # scalar elementwise only
```

When enabled, `add_nr` / `sub_nr` / `mult_nr` / `negate_nr` (and thus recording `+`, `-`, `*`, unary `-`) use vDSP for floating dtypes, including scalar↔vector broadcast. Inputs are materialized with `contiguous()` inside the kernel; results are always contiguous.

#### Benchmark: memory-bound vs compute-bound

`main.cpp` prints two tables. The first times single elementwise ops
(`add`/`sub`/`mul`/`negate`) and shows only ~1.1–1.2× from SIMD — these ops are
**memory-bound**: each one moves ~3 arrays through DRAM but does ≤1 FLOP per
element, so DRAM bandwidth is the wall and wider math units don't help.

The second table (`fused_poly_loop` vs `fused_poly_simd`) evaluates a
degree-`P` polynomial per element with Horner's rule. Memory traffic is fixed
(read `x`, write `out` = 8 B/elem) while the FMA count — and thus **arithmetic
intensity** (FLOP/byte) — grows with `P`. As intensity rises the kernel becomes
**compute-bound** and the SIMD path pulls away, reaching ~15× on Apple Silicon:

```text
operation                loop ms     simd ms    speedup  flop/byte
poly deg 1                 4.778       1.762      2.71x        0.25
poly deg 4                18.938       1.469     12.89x        1.00
poly deg 16               57.936       4.001     14.48x        4.00
poly deg 64              362.996      22.869     15.87x       16.00
```

The scalar loop uses `#pragma clang loop vectorize(disable)` so it stays a true
one-lane loop; the SIMD path uses `vDSP_vpoly` (or an auto-vectorized loop when
`MANGO_ENABLE_VDSP=OFF`). Takeaway: SIMD only pays off once you feed it enough
math per byte to escape the bandwidth roofline.

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

**Consumer counting:** If the same intermediate is used twice, its node waits until both paths deliver gradients before running `backwardPass`. Leaves just accumulate. Must rebuild the graph each training step (one `backward()` per forward).

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
