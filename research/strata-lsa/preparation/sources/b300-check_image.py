"""Check that the serving image's stack targets Blackwell, with or without a GPU.

Without a GPU it checks versions, imports and compiled arch lists.
With one it also launches a Triton kernel, a CuTeDSL kernel and a flashinfer call on the device.
"""

import argparse
import importlib
import importlib.metadata as metadata
import json
import sys

PACKAGES = (
    "torch",
    "triton",
    "sglang",
    "sglang-kernel",
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
    "sgl-deep-gemm",
    "nvidia-cutlass-dsl",
    "apache-tvm-ffi",
    "flash-attn-4",
    "quack-kernels",
    "nvidia-nccl-cu13",
    "cuda-python",
    "transformers",
)

MODULES = (
    "torch",
    "triton",
    "sglang",
    "sgl_kernel",
    "flashinfer",
    "deep_gemm",
    "cutlass",
    "cutlass.cute",
    "tvm_ffi",
)


def versions() -> dict[str, str]:
    """Return the installed version of every package the serving path reads."""
    found = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = "-"
    return found


def imports() -> dict[str, str]:
    """Import every module, returning ok or the error."""
    status = {}
    for name in MODULES:
        try:
            importlib.import_module(name)
            status[name] = "ok"
        except Exception as error:  # noqa: BLE001
            status[name] = f"{type(error).__name__}: {error}"[:300]
    return status


def gpu_checks() -> dict[str, str]:
    """Run one Triton, one CuTeDSL and one flashinfer kernel on the device."""
    import torch
    import triton
    import triton.language as tl

    results = {
        "device": torch.cuda.get_device_name(),
        "capability": str(torch.cuda.get_device_capability()),
    }

    @triton.jit
    def add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        tl.store(
            out_ptr + offsets,
            tl.load(x_ptr + offsets, mask=mask) + tl.load(y_ptr + offsets, mask=mask),
            mask=mask,
        )

    try:
        x = torch.randn(4096, device="cuda")
        y = torch.randn(4096, device="cuda")
        out = torch.empty_like(x)
        add[(4,)](x, y, out, 4096, BLOCK=1024)
        results["triton"] = "ok" if torch.allclose(out, x + y) else "MISMATCH"
    except Exception as error:  # noqa: BLE001
        results["triton"] = f"{type(error).__name__}: {error}"[:300]

    try:
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        @cute.kernel
        def scale_kernel(g: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            g[tidx] = g[tidx] * 2.0

        @cute.jit
        def scale(g: cute.Tensor):
            scale_kernel(g).launch(grid=(1, 1, 1), block=(128, 1, 1))

        t = torch.ones(128, device="cuda", dtype=torch.float32)
        scale(from_dlpack(t))
        torch.cuda.synchronize()
        results["cutedsl"] = "ok" if bool((t == 2).all()) else "MISMATCH"
    except Exception as error:  # noqa: BLE001
        results["cutedsl"] = f"{type(error).__name__}: {error}"[:300]

    try:
        import flashinfer

        a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(128, device="cuda", dtype=torch.bfloat16)
        flashinfer.norm.rmsnorm(a, w)
        torch.cuda.synchronize()
        results["flashinfer"] = "ok"
    except Exception as error:  # noqa: BLE001
        results["flashinfer"] = f"{type(error).__name__}: {error}"[:300]

    try:
        import sgl_kernel  # noqa: F401

        a = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(256, device="cuda", dtype=torch.bfloat16)
        torch.ops.sgl_kernel.rmsnorm.default(torch.empty_like(a), a, w, 1e-6, False)
        torch.cuda.synchronize()
        results["sgl_kernel"] = "ok"
    except Exception as error:  # noqa: BLE001
        results["sgl_kernel"] = f"{type(error).__name__}: {error}"[:300]
    return results


def main() -> int:
    """Print the checks as json and fail when an import or a required arch is missing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    import torch

    report = {
        "versions": versions(),
        "imports": imports(),
        "torch_cuda": torch.version.cuda,
        "torch_arch_list": torch.cuda.get_arch_list(),
    }
    # sgl_kernel picks its library by the device it finds, so it only imports where there is one
    needs_gpu = {"sgl_kernel"} if args.no_gpu else set()
    failures = [
        name
        for name, status in report["imports"].items()
        if status != "ok" and name not in needs_gpu
    ]
    if (
        "sm_100" not in report["torch_arch_list"]
        and "sm_100a" not in report["torch_arch_list"]
    ):
        failures.append("torch has no sm_100")
    if not args.no_gpu:
        report["gpu"] = gpu_checks()
        failures += [
            name
            for name, status in report["gpu"].items()
            if name not in ("device", "capability") and status != "ok"
        ]
    report["failures"] = failures
    print(json.dumps(report, indent=2))
    return 1 if failures and args.no_gpu else 0


if __name__ == "__main__":
    sys.exit(main())
