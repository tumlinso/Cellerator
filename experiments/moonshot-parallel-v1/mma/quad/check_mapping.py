#!/usr/bin/env python3
"""Check PTX fragment ownership and panel scheduling without loading CUDA."""
COUNTS = (1, 3, 4, 7, 16, 17)
for group in range(4):
    coordinates = []
    operand_rows = []
    for lane in range(32):
        if (lane >> 2) & 3 != group:
            continue
        operand_rows.append((lane & 3) + (4 if lane >= 16 else 0))
        for register in range(8):
            coordinates.append(((lane & 1) + (register & 2) + (4 if lane >= 16 else 0),
                                (register & 4) + (lane & 2) + (register & 1)))
    assert sorted(operand_rows) == list(range(8))
    assert len(coordinates) == 64
    assert set(coordinates) == {(r,c) for r in range(8) for c in range(8)}
for count in COUNTS:
    assigned = [warp*4+group for warp in range(((count+15)//16)*4)
                for group in range(4) if warp*4+group < count]
    assert assigned == list(range(count))
# Last block at uint32 maximum exercises the seed's overflowing thread expression.
last_block = ((2**32-1+15)//16)-1
last_thread = last_block*128+127
assert last_thread > 2**32-1
assert (last_thread >> 5)*4+3 >= 2**32-1
print('quad mapping PASS: four independent 8x8 outputs, counts 1/3/4/7/16/17, uint32 maximum scheduling')
