# XNNPACK Tutorial

## Installation

To get started, clone this repository along with its submodules:

```bash
git clone git@github.com:HaroldBenoit/xnnpack-tutorial.git
cd xnnpack-tutorial
git submodule update --init --recursive
```

This will ensure the XNNPACK submodule is properly initialized and checked out.

Then, let's build XNNPACK:

```bash
cd XNNPACK
./scripts/build-local.sh
```

Finally, we can test our example SwiGLU kernel by compiling it with `./build_swiglu.sh` and then running it with `./minimal_swiglu_kernel`

You should get this output:

```
Output: [24.203243, 52.594986]
```


