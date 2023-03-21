from disent.dataset.data import XYObjectData

data = XYObjectData(grid_size=4, min_square_size=1, max_square_size=2, square_size_spacing=1, palette="rgb_1")

print(f"Number of observations: {len(data)} == {data.size}")
print(f"Observation shape: {data.img_shape}")
print(f"Num Factors: {data.num_factors}")
print(f"Factor Names: {data.factor_names}")
print(f"Factor Sizes: {data.factor_sizes}")

for i, obs in enumerate(data):
    print(
        f"i={i}",
        f'pos: ({", ".join(data.factor_names)}) = {tuple(data.idx_to_pos(i))}',
        f"obs={obs.tolist()}",
        sep=" | ",
    )
