from disent.data.groundtruth import XYSquaresData

data = XYSquaresData(square_size=1, image_size=2, num_squares=2)

print(f'Number of observations: {len(data)} == {data.size}')
print(f'Observation shape: {data.observation_shape}')
print(f'Num Factors: {data.num_factors}')
print(f'Factor Names: {data.factor_names}')
print(f'Factor Sizes: {data.factor_sizes}')

for i, obs in enumerate(data):
    print(
        f'i={i}',
        f'pos: ({", ".join(data.factor_names)}) = {tuple(data.idx_to_pos(i))}',
        f'obs={obs.tolist()}',
        sep=' | ',
    )