import math
import time
import numpy as np
from tqdm import tqdm

"""
Utilities for converting and testing different chunk sizes of hdf5 files
"""


# ========================================================================= #
# hdf5                                                                   #
# ========================================================================= #


def bytes_to_human(size_bytes, decimals=3, color=True):
    if size_bytes == 0:
        return "0B"
    size_name = ("B  ", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")
    size_color = (None,     92,   93,    91,    91,    91,    91,    91,    91)
    i = int(math.floor(math.log(size_bytes, 1024)))
    s = round(size_bytes / math.pow(1024, i), decimals)
    name = f'\033[{size_color[i]}m{size_name[i]}\033[0m' if color else size_name[i]
    return f"{s:{4+decimals}.{decimals}f} {name}"


def hdf5_print_entry_data_stats(data, dataset, label='STATISTICS', print_mode='all'):
    dtype = data[dataset].dtype
    itemsize = data[dataset].dtype.itemsize
    # chunk
    chunks = np.array(data[dataset].chunks)
    data_per_chunk = np.prod(chunks) * itemsize
    # entry
    shape = np.array([1, *data[dataset].shape[1:]])
    data_per_entry = np.prod(shape) * itemsize
    # chunks per entry
    chunks_per_dim = np.ceil(shape / chunks).astype('int')
    chunks_per_entry = np.prod(chunks_per_dim)
    read_data_per_entry = data_per_chunk * chunks_per_entry
    # print info
    if print_mode == 'all':
        if label:
            tqdm.write(f'[{label}]: \033[92m{dataset}\033[0m')
        tqdm.write(
            f'\t\033[90mentry shape:\033[0m      {str(list(shape)):18s} \033[93m{bytes_to_human(data_per_entry)}\033[0m'
            f'\n\t\033[90mchunk shape:\033[0m      {str(list(chunks)):18s} \033[93m{bytes_to_human(data_per_chunk)}\033[0m'
            f'\n\t\033[90mchunks per entry:\033[0m {str(list(chunks_per_dim)):18s} \033[93m{bytes_to_human(read_data_per_entry)}\033[0m (\033[91m{chunks_per_entry}\033[0m)'
        )
    elif print_mode == 'minimal':
        tqdm.write(
            f'[{label:3s}] entry: {str(list(shape)):18s} ({str(dtype):8s}) \033[93m{bytes_to_human(data_per_entry)}\033[0m chunk: {str(list(chunks)):18s} \033[93m{bytes_to_human(data_per_chunk)}\033[0m chunks per entry: {str(list(chunks_per_dim)):18s} \033[93m{bytes_to_human(read_data_per_entry)}\033[0m (\033[91m{chunks_per_entry}\033[0m)'
        )

def hd5f_print_dataset_info(data, dataset, label='DATASET'):
    if label:
        tqdm.write(f'[{label}]: \033[92m{dataset}\033[0m')
    tqdm.write(
          f'\t\033[90mraw:\033[0m                {data[dataset]}'
          f'\n\t\033[90mchunks:\033[0m           {data[dataset].chunks}'
          f'\n\t\033[90mcompression:\033[0m      {data[dataset].compression}'
          f'\n\t\033[90mcompression lvl:\033[0m  {data[dataset].compression_opts}'
    )


def hdf5_resave_dataset(inp_data, out_data, dataset, chunks=None, compression=None, compression_opts=None, batch_size=None, max_entries=None, dry_run=False, print_mode='minimal'):
    # print_dataset_info(inp_data, dataset, label='INPUT')
    # create new dataset
    out_data.create_dataset(
        name=dataset,
        shape=inp_data[dataset].shape,
        dtype=inp_data[dataset].dtype,
        chunks=chunks,
        compression=compression,
        compression_opts=compression_opts
    )

    hdf5_print_entry_data_stats(inp_data, dataset, label=f'IN', print_mode=print_mode)
    hdf5_print_entry_data_stats(out_data, dataset, label=f'OUT', print_mode=print_mode)
    tqdm.write('')

    if not dry_run:
        # choose chunk size
        if batch_size is None:
            batch_size = inp_data[dataset].chunks[0]
        # batched copy
        entries = len(inp_data[dataset])
        with tqdm(total=entries) as progress:
            for i in range(0, max_entries if max_entries else entries, batch_size):
                out_data[dataset][i:i + batch_size] = inp_data[dataset][i:i + batch_size]
                progress.update(batch_size)
        tqdm.write('')


def hdf5_test_entries_per_second(data, dataset, access_method='random', max_entries=48000, timeout=10):
    # num entries to test
    n = min(len(data[dataset]), max_entries)

    # get access method
    if access_method == 'sequential':
        indices = np.arange(n)
    elif access_method == 'random':
        indices = np.arange(n)
        np.random.shuffle(indices)
    else:
        raise KeyError('Invalid access method')

    # iterate through dataset, exit on timeout or max_entries
    start_time = time.time()
    for i, idx in enumerate(indices):
        entry = data[dataset][idx]
        if time.time() - start_time > timeout or i >= max_entries:
            break

    # calculate score
    entries_per_sec = (i + 1) / (time.time() - start_time)
    return entries_per_sec

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
