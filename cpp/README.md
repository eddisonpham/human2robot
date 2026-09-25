# C++ trajectory subsystem (h2r_traj)

Standalone C++20 library for trajectory validation, projection, and
optimization over retargeted robot demonstrations, plus pybind11 bindings
consumed by `src/human2robot/cpp_bindings/`.

Spec: `docs/HANDOFF_RESPONSE.md` sections 3, 5, 14, 17, 18, 19.

## Layout

```text
cpp/
  include/h2r/     Public headers (trajectory, interpolation, derivatives,
                   smoothing, constraints, robot_model, analytic_chain,
                   collision_checker, metrics, cost, optimizer, csv)
  src/             Implementation, mirroring include layout
  tests/           GoogleTest suites (57 tests)
  examples/        optimize_demo: CLI over CSV files
  src/bindings/    pybind11 module h2r_cpp
```

## Toolchain (this machine)

- clang++ 22.1.6 targeting MSVC (BuildTools 14.50 under Program Files (x86)),
  no g++, no direct cl usage needed
- Ninja 1.13 installed into the project venv via uv (`uv add ninja`)
- CMake 4.1.3 on PATH
- Eigen 3.4.0, GoogleTest 1.14, pybind11 2.13.6 fetched by FetchContent

## Build and test

```bash
export PATH="$PWD/.venv/Scripts:$PATH"
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_MAKE_PROGRAM="$PWD/.venv/Scripts/ninja.exe" \
  -DPYTHON_EXECUTABLE="$PWD/.venv/Scripts/python.exe"
cmake --build cpp/build -j 16
cd cpp/build && ctest --output-on-failure
```

The pybind11 module is named `h2r_cpp.cp311-win_amd64.pyd` and is picked up
automatically by `src/human2robot/cpp_bindings/`. Set `PYTHON_EXECUTABLE`
to the project venv or the module targets the wrong interpreter.

## Status

Phases B-E complete and validated: trajectory representation, interpolation,
finite differences, smoothing, limit projection, metrics, weighted cost,
stochastic optimizer with projection. 57/57 C++ tests pass. Python
integration tests cover numpy roundtrip, equivalence against numpy
references, determinism, and MuJoCo replay of optimized demos.

Measured on T=200, dof=22 (release build): optimizer 0.33 ms per iteration,
smoothing 13.7x faster than the equivalent numpy loop.
