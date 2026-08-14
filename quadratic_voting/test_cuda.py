def main() -> None:
    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot access CUDA; verify that the NVIDIA driver is "
            "installed and that the installed PyTorch build supports CUDA."
        )

    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    print(f"Native CUDA BF16 support: {native_bf16}")
    if not native_bf16:
        raise RuntimeError(
            "The CUDA device does not report native BF16 support required by "
            "the current Gemma checkpoint; use a compatible GPU or explicitly "
            "validate a different model dtype."
        )

    device = torch.device("cuda")
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=torch.bfloat16)
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device, dtype=torch.bfloat16)
    result = left @ right
    expected = torch.tensor(
        [[19.0, 22.0], [43.0, 50.0]], device=device, dtype=torch.bfloat16
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(result, expected)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Result: {result.cpu().tolist()}")


if __name__ == "__main__":
    main()
