import torch


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot access CUDA; verify the NVIDIA driver and that "
            "the Nix development shell exposes the host driver libraries."
        )

    device = torch.device("cuda")
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
    result = left @ right
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], device=device)

    torch.cuda.synchronize()
    torch.testing.assert_close(result, expected)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Result: {result.cpu().tolist()}")


if __name__ == "__main__":
    main()
