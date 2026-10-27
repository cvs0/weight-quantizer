import torch

from ptqkit.pack import pack_int4, unpack_int4


def test_nibble_layout() -> None:
    codes = torch.tensor([[1, 2, 15, 0], [0, 15, 8, 8]], dtype=torch.int32)
    packed = pack_int4(codes)
    assert int(packed[0, 0].item()) == 0x21
    assert int(packed[0, 1].item()) == 0x0F
    assert int(packed[1, 0].item()) == 0xF0
    restored = unpack_int4(packed, 4)
    assert torch.equal(restored, codes.to(torch.uint8))


def test_odd_width_roundtrip() -> None:
    codes = torch.randint(0, 16, (5, 7), dtype=torch.int32)
    restored = unpack_int4(pack_int4(codes), 7)
    assert torch.equal(restored, codes.to(torch.uint8))
    assert pack_int4(codes).shape == (5, 4)


def test_random_roundtrip() -> None:
    codes = torch.randint(0, 16, (13, 128), dtype=torch.int32)
    assert torch.equal(unpack_int4(pack_int4(codes), 128), codes.to(torch.uint8))
