import sys
import torch
from modules import ScreeningEncoder

NUM_FACTORS  = 15
ENCODING_DIM = 32
HIDDEN_DIM   = 128
N            = 200


def test_encoder_batch():
    enc = ScreeningEncoder(NUM_FACTORS, ENCODING_DIM, HIDDEN_DIM)
    enc.eval()

    xi = torch.randn(N, NUM_FACTORS)
    y  = torch.randn(N)

    with torch.no_grad():
        out = enc(xi, y)

    assert out.shape == torch.Size([N, ENCODING_DIM]), (
        f"Shape is {list(out.shape)}, expected [{N}, {ENCODING_DIM}]. "
        f"y is likely being collapsed to a single element."
    )

    xi2 = torch.randn(2, NUM_FACTORS)
    y2  = torch.tensor([+10.0, -10.0])
    with torch.no_grad():
        out2 = enc(xi2, y2)

    assert not torch.allclose(out2[0], out2[1], atol=1e-4), (
        "Both rollouts produced identical embeddings despite different y values. "
        "y collapse bug is still present."
    )

    print("PASSED — encoder preserves batch dim and uses each rollout's own y.")


if __name__ == "__main__":
    try:
        test_encoder_batch()
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED — {e}")
        sys.exit(1)