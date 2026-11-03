import pickle
import pytest
import torch

from lewm.preprocessing import(
    IMAGENET_MEAN,
    IMAGENET_STD,
    ZScoreNormalizer,
    build_zscore_normalizer,
    preprocess_pixels,
)

def test_output_shape_with_leading_dims():
    pixels = torch.randint(0, 256, (2, 3, 28, 28, 3), dtype=torch.uint8)
    out = preprocess_pixels(pixels, image_size=56)
    assert out.shape == (2, 3, 3, 56, 56)
    assert out.dtype == torch.float32

def test_rejects_non_uint8():
    with pytest.raises(ValueError):
        preprocess_pixels(torch.zeros(1, 8, 8, 3), image_size=8)

def test_constant_images_get_imagenet_normalization():
    white = torch.full((1, 8, 8, 3), 255, dtype=torch.uint8)
    black = torch.zeros(1, 8, 8, 3, dtype=torch.uint8)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    out_white = preprocess_pixels(white, image_size=8)[0]
    out_black = preprocess_pixels(black, image_size=8)[0]

    assert torch.allclose(out_white, ((1.0 - mean) / std).expand(3, 8, 8), atol=1e-5)
    assert torch.allclose(out_black, (-mean / std).expand(3, 8, 8), atol=1e-5)

def test_matches_torchvision_pipeline():
    v2 = pytest.importorskip("torchvision.transforms.v2")
    pixels = torch.randint(0, 256, (4, 28, 28, 3), dtype=torch.uint8)

    ours = preprocess_pixels(pixels, image_size=56)

    chw = pixels.permute(0, 3, 1, 2)
    expected = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            v2.Resize(56, antialias=True),
        ]
    )(chw)

    assert torch.allclose(ours, expected, atol=1e-5)

def test_zscore_ignores_nan_rows_for_statistics():
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [float("nan"), 100.0]])
    normalizer = build_zscore_normalizer(data.numpy())

    assert torch.allclose(normalizer.mean, torch.tensor([[2.0, 3.0]]))
    # sample std (N - 1), like torch's default
    assert torch.allclose(normalizer.std, data[:2].std(0, keepdim=True))

def test_zscore_output_is_standardized_float32():
    data = torch.randn(1000, 3) * 5 + 2
    normalizer = build_zscore_normalizer(data.numpy())
    out = normalizer(data)

    assert out.dtype == torch.float32
    assert torch.allclose(out.mean(0), torch.zeros(3), atol=1e-4)
    assert torch.allclose(out.std(0), torch.ones(3), atol=1e-4)

def test_zscore_normalizer_is_picklable():
    normalizer = ZScoreNormalizer(torch.zeros(1, 2), torch.ones(1, 2))
    restored = pickle.loads(pickle.dumps(normalizer))
    x = torch.randn(4, 2)
    assert torch.equal(restored(x), normalizer(x))
