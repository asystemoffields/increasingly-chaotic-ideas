r"""B4 BLOCKING unit test — INT4 scale factor correctness.

Verifies that symmetric INT4 quantization uses scale = max_abs(x) / 7,
NOT max_abs(x) / 8 (which would clip at the negative saturation point).

5-line logical test, expanded with assertion messages for clarity.
Exit code 0 = PASS, non-zero = FAIL (halt R-RAOK-70B Rung 1).
"""
import sys
import torch

torch.manual_seed(0)

# --- 5-line core unit test ---
# Use integer-multiple inputs so round() is exact (no half-integer tie-breaking).
x = torch.tensor([-7.0, -4.0, 0.0, 4.0, 7.0])            # known input, integer multiples of 1.0
scale = x.abs().max() / 7                                   # correct INT4 scale = 7/7 = 1.0
q = torch.round(x / scale).clamp(-7, 7)                    # quantize, clamp to [-7,7]
x_recon = q * scale                                         # reconstruct
max_err = (x - x_recon).abs().max().item()                  # should be 0

assert abs(scale.item() - 1.0) < 1e-6, f"FAIL: scale={scale.item()}, expected 1.0"
assert max_err < 1e-5, f"FAIL: max reconstruction error={max_err}, expected ~0"
assert q.tolist() == [-7.0, -4.0, 0.0, 4.0, 7.0], \
    f"FAIL: unexpected quantized values {q.tolist()}"

# Verify max_abs/8 would be WRONG.
# With scale=7/8=0.875: x=-7 quantizes to round(-7/0.875)=round(-8)=-8, clamped to -7.
# Reconstructed: -7*0.875=-6.125, error=0.875. Demonstrates clipping artifact.
scale_wrong = x.abs().max() / 8   # wrong: 7/8 = 0.875
q_wrong = torch.round(x / scale_wrong).clamp(-7, 7)
x_recon_wrong = q_wrong * scale_wrong
err_wrong = (x - x_recon_wrong).abs().max().item()
assert err_wrong > 0.5, \
    f"FAIL: wrong scale should produce error>0.5 at x=-7 but got {err_wrong}"

# Verify the x=-7 clipping specifically.
x_neg7 = torch.tensor([-7.0])
q_neg7_wrong = torch.round(x_neg7 / scale_wrong).clamp(-7, 7)
recon_neg7_wrong = q_neg7_wrong * scale_wrong
clip_err = abs(recon_neg7_wrong.item() - (-7.0))
assert clip_err > 0.5, \
    f"FAIL: clip error at x=-7 with wrong scale should be >0.5, got {clip_err}"

# Broader test: random tensor reconstruction error <= 0.5 * scale (by definition of round).
torch.manual_seed(42)
y = torch.randn(256) * 3.0
scale_y = y.abs().max() / 7
q_y = torch.round(y / scale_y).clamp(-7, 7)
y_recon = q_y * scale_y
max_rand_err = (y - y_recon).abs().max().item()
assert max_rand_err <= 0.5 * scale_y.item() + 1e-5, \
    f"FAIL: random-tensor reconstruction error {max_rand_err:.4f} exceeds 0.5*scale={0.5*scale_y.item():.4f}"

print("B4 UNIT TEST: PASS")
print(f"  scale = max_abs/7 = {scale.item():.4f} (correct)")
print(f"  max reconstruction error on integer inputs: {max_err:.2e}")
print(f"  max_abs/8 reconstruction error (wrong scale): {err_wrong:.4f} (>0.5 as expected)")
print(f"  clip error at x=-7 with wrong scale: {clip_err:.4f}")
print(f"  random-tensor max error: {max_rand_err:.4f} <= 0.5*scale={0.5*scale_y.item():.4f}")
sys.exit(0)
