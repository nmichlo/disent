import os
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from disent.dataset.util.hdf5 import bytes_to_human, hdf5_resave_dataset, hdf5_test_entries_per_second

# ========================================================================= #
# Shapes3d Utilities                                                        #
# ========================================================================= #


def hdf5_3dshapes_resave(
        load_name=None,
        save_name=None,
        compression='gzip',
        compression_lvl=4,
        image_chunks=32,
        image_block_size=32,
        image_channels=1,
        label_chunks=4096,
        stop_early=False,
        dry_run=False,
        test=False,
):
    if load_name is None:
        load_name = f'data/dataset/3dshapes.h5'
    if save_name is None:
        save_name = f'data/dataset/3dshapes_{compression}{compression_lvl}_{image_chunks}-{image_block_size}-{image_channels}_{label_chunks}{"_dry" if dry_run else ""}'

    with h5py.File(load_name, 'r') as inp_data:
        with h5py.File(save_name, 'w') as out_data:
            hdf5_resave_dataset(inp_data, out_data, 'images', (image_chunks, image_block_size, image_block_size, image_channels), compression, compression_lvl, max_entries=48000 if stop_early else None, dry_run=dry_run)
            # hdf5_resave_dataset(inp_data, out_data, 'labels', (label_chunks, 6), compression, compression_lvl, max_entries=48000 if stop_early else None, dry_run=dry_run)
            # File Size:
            print(f'[FILE SIZES] IN: {bytes_to_human(os.path.getsize(load_name))} OUT: {bytes_to_human(os.path.getsize(save_name))}\n')
            if test:
                entries_per_sec = hdf5_test_entries_per_second(out_data, 'images')
                tqdm.write(f'\tTEST: entries per second = {entries_per_sec:.2f}')
                os.rename(save_name, os.path.splitext(save_name)[0] + f'__{entries_per_sec:.2f}eps.h5')
                tqdm.write('')


def get_info_from_filenames(dir_path='data/dataset'):
    FILE_DATA = []

    for path in sorted(os.listdir(dir_path)):
        try:
            # extract components
            file_name, ext = os.path.splitext(path)
            name, comp, img, lbl_chunks, _, eps = file_name.split('_')

            # parse values

            compression, compression_lvl = comp[:4], comp[4:]
            try:
                compression_lvl = int(compression_lvl)
            except:
                pass

            img_chunks, img_block, img_channels = [int(v) for v in img.split('-')]
            lbl_chunks = int(lbl_chunks)
            eps = float(eps[:-3])

            # chunk
            chunks = np.array([img_chunks, img_block, img_block, img_channels])
            data_per_chunk = np.prod(chunks)
            # entry
            shape = np.array([1, 64, 64, 3])
            data_per_entry = np.prod(shape)
            # chunks per entry
            chunks_per_dim = np.ceil(shape / chunks).astype('int')
            chunks_per_entry = np.prod(chunks_per_dim)
            read_data_per_entry = data_per_chunk * chunks_per_entry

            FILE_DATA.append(dict(
                path=path,
                compression=compression,
                compression_lvl=compression_lvl,
                img_chunks=img_chunks,
                img_block=img_block,
                img_channels=img_channels,
                lbl_chunks=lbl_chunks,
                eps=eps,
                data_per_entry=data_per_entry,
                data_per_chunk=data_per_chunk,
                read_data_per_entry=read_data_per_entry,
                file_size=os.path.getsize('data/dataset/' + path),
            ))

        except:
            pass

    return FILE_DATA

def print_info(file_data, sort_keys=('file_size', 'path')):
    for dat in sorted(file_data, key=lambda x: tuple([x[k] for k in sort_keys])):
        print(f'[{bytes_to_human(dat["file_size"])}|{bytes_to_human(dat["file_size"]*10)}] {dat["path"]:45s} | {dat["eps"]:7.2f} | {bytes_to_human(dat["read_data_per_entry"])} | {bytes_to_human(dat["data_per_chunk"])}')


# ========================================================================= #
# Main                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    # resave(image_chunks=15000, image_block_size=4, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=5000, image_block_size=4, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=3750, image_block_size=8, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=1250, image_block_size=8, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=937, image_block_size=16, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=312, image_block_size=16, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=234, image_block_size=32, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=78, image_block_size=32, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=58, image_block_size=64, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=19, image_block_size=64, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)

    # resave(image_chunks=128, image_block_size=4, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=128, image_block_size=4, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=128, image_block_size=8, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=128, image_block_size=8, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=128, image_block_size=16, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=128, image_block_size=16, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=128, image_block_size=32, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=128, image_block_size=32, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=128, image_block_size=64, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=128, image_block_size=64, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)

    # resave(image_chunks=32, image_block_size=4, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=32, image_block_size=4, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=32, image_block_size=8, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=32, image_block_size=8, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=32, image_block_size=16, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=32, image_block_size=16, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=32, image_block_size=32, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=32, image_block_size=32, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=32, image_block_size=64, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=32, image_block_size=64, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)

    # resave(image_chunks=256, image_block_size=4, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=256, image_block_size=4, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=256, image_block_size=8, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=256, image_block_size=8, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=256, image_block_size=16, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=256, image_block_size=16, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=256, image_block_size=32, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=256, image_block_size=32, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=256, image_block_size=64, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=256, image_block_size=64, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)

    # resave(image_chunks=64, image_block_size=4, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=64, image_block_size=4, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=64, image_block_size=8, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=64, image_block_size=8, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=64, image_block_size=16, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=64, image_block_size=16, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=64, image_block_size=32, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=64, image_block_size=32, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=64, image_block_size=64, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=64, image_block_size=64, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)

    # # resave(image_chunks=256, image_block_size=4, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # # resave(image_chunks=256, image_block_size=4, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # # resave(image_chunks=64, image_block_size=8, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # # resave(image_chunks=64, image_block_size=8, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=16, image_block_size=16, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=16, image_block_size=16, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=4, image_block_size=32, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=4, image_block_size=32, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=1, image_block_size=64, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=1, image_block_size=64, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    #
    # resave(image_chunks=4096, image_block_size=4, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=4096, image_block_size=4, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=1024, image_block_size=8, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=1024, image_block_size=8, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # # resave(image_chunks=256, image_block_size=16, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # # resave(image_chunks=256, image_block_size=16, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # # resave(image_chunks=64, image_block_size=32, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # # resave(image_chunks=64, image_block_size=32, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=16, image_block_size=64, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=16, image_block_size=64, image_channels=3, label_chunks=4096, stop_early=True, dry_run=False, test=True)

    # NOT ANALISED

    # resave(image_chunks=96, image_block_size=16, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=64, image_block_size=16, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=32, image_block_size=16, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    #
    # resave(image_chunks=96, image_block_size=8, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=64, image_block_size=8, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=32, image_block_size=8, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    #
    # resave(image_chunks=96, image_block_size=4, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=64, image_block_size=4, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)
    # resave(image_chunks=32, image_block_size=4, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)

    # ======================= #
    # Analyse Saved Files     #
    # ======================= #

    # [ 29.285 MiB] 3dshapes_gzip4_15000-4-1_4096__3.23eps.h5     |    3.23 | 175.781 MiB | 234.375 KiB
    # [ 34.649 MiB] 3dshapes_gzip4_4096-4-1_4096__10.64eps.h5     |   10.64 |  48.000 MiB |  64.000 KiB
    # [ 38.329 MiB] 3dshapes_gzip4_3750-8-1_4096__12.28eps.h5     |   12.28 |  43.945 MiB | 234.375 KiB
    # [ 50.058 MiB] 3dshapes_gzip4_1024-8-1_4096__40.65eps.h5     |   40.65 |  12.000 MiB |  64.000 KiB
    # [ 62.825 MiB] 3dshapes_gzip4_5000-4-3_4096__9.82eps.h5      |    9.82 |  58.594 MiB | 234.375 KiB
    # [ 63.273 MiB] 3dshapes_gzip4_4096-4-3_4096__12.26eps.h5     |   12.26 |  48.000 MiB | 192.000 KiB
    # [ 68.782 MiB] 3dshapes_gzip4_937-16-1_4096__45.23eps.h5     |   45.23 |  10.980 MiB | 234.250 KiB
    # [ 73.117 MiB] 3dshapes_gzip4_256-8-1_4096__113.47eps.h5     |  113.47 |   3.000 MiB |  16.000 KiB
    #*[ 74.613 MiB] 3dshapes_gzip4_256-16-1_4096__150.57eps.h5    |  150.57 |   3.000 MiB |  64.000 KiB
    # [ 79.664 MiB] 3dshapes_gzip4_1250-8-3_4096__36.40eps.h5     |   36.40 |  14.648 MiB | 234.375 KiB
    # [ 80.189 MiB] 3dshapes_gzip4_1024-8-3_4096__45.27eps.h5     |   45.27 |  12.000 MiB | 192.000 KiB
    # [ 81.632 MiB] 3dshapes_gzip4_256-4-1_4096__62.46eps.h5      |   62.46 |   3.000 MiB |   4.000 KiB
    #*[ 82.435 MiB] 3dshapes_gzip4_128-16-1_4096__254.62eps.h5    |  254.62 |   1.500 MiB |  32.000 KiB
    # [ 84.383 MiB] 3dshapes_gzip4_128-8-1_4096__168.81eps.h5     |  168.81 |   1.500 MiB |   8.000 KiB
    # [ 84.891 MiB] 3dshapes_gzip4_256-4-3_4096__115.27eps.h5     |  115.27 |   3.000 MiB |  12.000 KiB
    # [ 87.198 MiB] 3dshapes_gzip4_256-32-1_4096__149.97eps.h5    |  149.97 |   3.000 MiB | 256.000 KiB
    # [ 87.513 MiB] 3dshapes_gzip4_256-8-3_4096__153.86eps.h5     |  153.86 |   3.000 MiB |  48.000 KiB
    # [ 87.825 MiB] 3dshapes_gzip4_234-32-1_4096__157.60eps.h5    |  157.60 |   2.742 MiB | 234.000 KiB
    # [ 93.644 MiB] 3dshapes_gzip4_128-32-1_4096__272.62eps.h5    |  272.62 |   1.500 MiB | 128.000 KiB
    # [ 96.920 MiB] 3dshapes_gzip4_128-8-3_4096__246.95eps.h5     |  246.95 |   1.500 MiB |  24.000 KiB
    #*[ 97.912 MiB] 3dshapes_gzip4_64-16-1_4096__1954.21eps.h5    | 1954.21 | 768.000 KiB |  16.000 KiB
    # [100.358 MiB] 3dshapes_gzip4_128-4-3_4096__154.85eps.h5     |  154.85 |   1.500 MiB |   6.000 KiB
    # [100.977 MiB] 3dshapes_gzip4_312-16-3_4096__131.42eps.h5    |  131.42 |   3.656 MiB | 234.000 KiB
    # [102.130 MiB] 3dshapes_gzip4_256-16-3_4096__155.43eps.h5    |  155.43 |   3.000 MiB | 192.000 KiB
    #*[106.542 MiB] 3dshapes_gzip4_64-32-1_4096__3027.06eps.h5    | 3027.06 | 768.000 KiB |  64.000 KiB
    # [106.696 MiB] 3dshapes_gzip4_64-8-1_4096__777.87eps.h5      |  777.87 | 768.000 KiB |   4.000 KiB
    # [108.600 MiB] 3dshapes_gzip4_128-16-3_4096__273.42eps.h5    |  273.42 |   1.500 MiB |  96.000 KiB
    # [112.515 MiB] 3dshapes_gzip4_128-4-1_4096__80.91eps.h5      |   80.91 |   1.500 MiB |   2.000 KiB
    # [115.242 MiB] 3dshapes_gzip4_64-8-3_4096__1855.37eps.h5     | 1855.37 | 768.000 KiB |  12.000 KiB
    #*[121.457 MiB] 3dshapes_gzip4_64-16-3_4096__3458.68eps.h5    | 3458.68 | 768.000 KiB |  48.000 KiB
    # [127.519 MiB] 3dshapes_gzip4_32-16-1_4096__1872.30eps.h5    | 1872.30 | 384.000 KiB |   8.000 KiB
    # [130.568 MiB] 3dshapes_gzip4_64-4-3_4096__643.22eps.h5      |  643.22 | 768.000 KiB |   3.000 KiB
    #*[131.429 MiB] 3dshapes_gzip4_32-32-1_4096__2976.50eps.h5    | 2976.50 | 384.000 KiB |  32.000 KiB
    # [146.867 MiB] 3dshapes_gzip4_32-16-3_4096__3354.44eps.h5    | 3354.44 | 384.000 KiB |  24.000 KiB
    # [149.890 MiB] 3dshapes_gzip4_32-8-3_4096__1772.73eps.h5     | 1772.73 | 384.000 KiB |   6.000 KiB
    # [156.784 MiB] 3dshapes_gzip4_32-8-1_4096__737.14eps.h5      |  737.14 | 384.000 KiB |   2.000 KiB
    # [165.268 MiB] 3dshapes_gzip4_64-4-1_4096__133.25eps.h5      |  133.25 | 768.000 KiB |   1.000 KiB
    # [173.404 MiB] 3dshapes_gzip4_256-64-1_4096__122.20eps.h5    |  122.20 |   3.000 MiB |   1.000 MiB
    # [173.655 MiB] 3dshapes_gzip4_128-64-1_4096__234.82eps.h5    |  234.82 |   1.500 MiB | 512.000 KiB
    # [174.163 MiB] 3dshapes_gzip4_64-64-1_4096__3734.94eps.h5    | 3734.94 | 768.000 KiB | 256.000 KiB
    # [174.303 MiB] 3dshapes_gzip4_58-64-1_4096__3768.85eps.h5    | 3768.85 | 696.000 KiB | 232.000 KiB
    #*[175.186 MiB] 3dshapes_gzip4_32-64-1_4096__3747.35eps.h5    | 3747.35 | 384.000 KiB | 128.000 KiB
    # [177.273 MiB] 3dshapes_gzip4_16-64-1_4096__3644.07eps.h5    | 3644.07 | 192.000 KiB |  64.000 KiB
    # [181.092 MiB] 3dshapes_gzip4_16-16-1_4096__1857.56eps.h5    | 1857.56 | 192.000 KiB |   4.000 KiB
    # [186.982 MiB] 3dshapes_gzip4_32-4-3_4096__620.46eps.h5      |  620.46 | 384.000 KiB |   1.500 KiB
    # [193.039 MiB] 3dshapes_gzip4_16-16-3_4096__3328.66eps.h5    | 3328.66 | 192.000 KiB |  12.000 KiB
    # [194.566 MiB] 3dshapes_gzip4_256-32-3_4096__127.59eps.h5    |  127.59 |   3.000 MiB | 768.000 KiB
    # [194.894 MiB] 3dshapes_gzip4_128-32-3_4096__240.13eps.h5    |  240.13 |   1.500 MiB | 384.000 KiB
    # [195.286 MiB] 3dshapes_gzip4_78-32-3_4096__4387.44eps.h5    | 4387.44 | 936.000 KiB | 234.000 KiB
    # [195.560 MiB] 3dshapes_gzip4_64-32-3_4096__4299.35eps.h5    | 4299.35 | 768.000 KiB | 192.000 KiB
    #*[196.880 MiB] 3dshapes_gzip4_32-32-3_4096__4300.62eps.h5    | 4300.62 | 384.000 KiB |  96.000 KiB
    # [199.859 MiB] 3dshapes_gzip4_4-32-1_4096__2675.23eps.h5     | 2675.23 |  48.000 KiB |   4.000 KiB
    # [201.911 MiB] 3dshapes_gzip4_256-64-3_4096__126.09eps.h5    |  126.09 |   3.000 MiB |   3.000 MiB
    # [202.034 MiB] 3dshapes_gzip4_128-64-3_4096__238.18eps.h5    |  238.18 |   1.500 MiB |   1.500 MiB
    # [202.272 MiB] 3dshapes_gzip4_64-64-3_4096__4729.32eps.h5    | 4729.32 | 768.000 KiB | 768.000 KiB
    #*[202.748 MiB] 3dshapes_gzip4_32-64-3_4096__4784.69eps.h5    | 4784.69 | 384.000 KiB | 384.000 KiB
    # [203.369 MiB] 3dshapes_gzip4_19-64-3_4096__4710.94eps.h5    | 4710.94 | 228.000 KiB | 228.000 KiB
    # [203.711 MiB] 3dshapes_gzip4_16-64-3_4096__4661.72eps.h5    | 4661.72 | 192.000 KiB | 192.000 KiB
    # [212.585 MiB] 3dshapes_gzip4_4-32-3_4096__4011.14eps.h5     | 4011.14 |  48.000 KiB |  12.000 KiB
    # [213.067 MiB] 3dshapes_gzip4_1-64-1_4096__3181.71eps.h5     | 3181.71 |  12.000 KiB |   4.000 KiB
    # [224.992 MiB] 3dshapes_gzip4_1-64-3_4096__4231.57eps.h5     | 4231.57 |  12.000 KiB |  12.000 KiB
    # [267.425 MiB] 3dshapes_gzip4_32-4-1_4096__144.90eps.h5      |  144.90 | 384.000 KiB | 512.000 B

    # ======================= #
    # Interesting Saved Files #
    # ======================= #

    # # # *[ 29.285 MiB] 3dshapes_gzip4_15000-4-1_4096__3.23eps.h5     |    3.23 | 175.781 MiB | 234.375 KiB
    # # resave(image_chunks=15000, image_block_size=4, image_channels=1, label_chunks=4096, stop_early=False, dry_run=False, test=True)
    #
    # # # *[ 74.613 MiB] 3dshapes_gzip4_256-16-1_4096__150.57eps.h5    |  150.57 |   3.000 MiB |  64.000 KiB
    # # resave(image_chunks=256, image_block_size=16, image_channels=1, label_chunks=4096, stop_early=False, dry_run=False, test=True)

    # [775.470 MiB] 3dshapes_gzip4_128-16-1_4096__247.32eps.h5    |  247.32 |   1.500 MiB |  32.000 KiB
    # *[ 82.435 MiB] 3dshapes_gzip4_128-16-1_4096__254.62eps.h5    |  254.62 |   1.500 MiB |  32.000 KiB
    # hdf5_3dshapes_resave(image_chunks=128, image_block_size=16, image_channels=1, label_chunks=4096, stop_early=False, dry_run=False, test=True)

    # # [917.514 MiB] 3dshapes_gzip4_64-16-1_4096__1861.96eps.h5    | 1861.96 | 768.000 KiB |  16.000 KiB
    # # *[ 97.912 MiB] 3dshapes_gzip4_64-16-1_4096__1954.21eps.h5    | 1954.21 | 768.000 KiB |  16.000 KiB
    # resave(image_chunks=64, image_block_size=16, image_channels=1, label_chunks=4096, stop_early=False, dry_run=False, test=True)

    # [1012.996 MiB] 3dshapes_gzip4_64-32-1_4096__3020.21eps.h5    | 3020.21 | 768.000 KiB |  64.000 KiB
    # *[106.542 MiB] 3dshapes_gzip4_64-32-1_4096__3027.06eps.h5    | 3027.06 | 768.000 KiB |  64.000 KiB
    # hdf5_3dshapes_resave(image_chunks=64, image_block_size=32, image_channels=1, label_chunks=4096, stop_early=False, dry_run=False, test=True)

    # # # *[121.457 MiB] 3dshapes_gzip4_64-16-3_4096__3458.68eps.h5    | 3458.68 | 768.000 KiB |  48.000 KiB
    # # resave(image_chunks=64, image_block_size=16, image_channels=3, label_chunks=4096, stop_early=False, dry_run=False, test=True)
    #
    # # # *[131.429 MiB] 3dshapes_gzip4_32-32-1_4096__2976.50eps.h5    | 2976.50 | 384.000 KiB |  32.000 KiB
    # # resave(image_chunks=32, image_block_size=32, image_channels=1, label_chunks=4096, stop_early=False, dry_run=False, test=True)
    #
    # # *[175.186 MiB] 3dshapes_gzip4_32-64-1_4096__3747.35eps.h5    | 3747.35 | 384.000 KiB | 128.000 KiB
    # resave(image_chunks=32, image_block_size=64, image_channels=1, label_chunks=4096, stop_early=False, dry_run=False, test=True)
    #
    # # [  1.620 GiB] 3dshapes_gzip4_32-64-1_4096__3663.23eps.h5    | 3663.23 | 384.000 KiB | 128.000 KiB
    # # # *[196.880 MiB] 3dshapes_gzip4_32-32-3_4096__4300.62eps.h5    | 4300.62 | 384.000 KiB |  96.000 KiB
    # # resave(image_chunks=32, image_block_size=32, image_channels=3, label_chunks=4096, stop_early=False, dry_run=False, test=True)

    # [  1.787 GiB] 3dshapes_gzip4_32-64-3_4096__4700.43eps.h5    | 4700.43 | 384.000 KiB | 384.000 KiB
    # *[202.748 MiB] 3dshapes_gzip4_32-64-3_4096__4784.69eps.h5    | 4784.69 | 384.000 KiB | 384.000 KiB
    # hdf5_3dshapes_resave(image_chunks=32, image_block_size=64, image_channels=3, label_chunks=4096, stop_early=False, dry_run=False, test=True)

    # =================== #
    # RANDOM ACCESS TESTS #
    # =================== #

    # OLD WITH LARGE CHUNKS:
    # [106.381 MiB|  1.039 GiB] 3dshapes_gzip4_64-32-1_4096__406.75eps.h5     |  406.75 | 768.000 KiB |  64.000 KiB
    # hdf5_3dshapes_resave(image_chunks=64, image_block_size=32, image_channels=1, label_chunks=4096, stop_early=True, dry_run=False, test=True)

    # NEW WITH ELEMENTS AS CHUNKS:
    # [204.065 MiB|  1.993 GiB] 3dshapes_gzip4_12-64-3_4096__1237.03eps.h5    | 1237.03 | 144.000 KiB | 144.000 KiB
    # [178.386 MiB|  1.742 GiB] 3dshapes_gzip4_12-64-1_4096__1129.64eps.h5    | 1129.64 | 144.000 KiB |  48.000 KiB
    # hdf5_3dshapes_resave(image_chunks=12, image_block_size=64, image_channels=3, label_chunks=4096, compression='gzip', compression_lvl=4, stop_early=True, dry_run=False, test=True)
    # hdf5_3dshapes_resave(image_chunks=12, image_block_size=64, image_channels=1, label_chunks=4096, compression='gzip', compression_lvl=4, stop_early=True, dry_run=False, test=True)

    # [210.778 MiB|  2.058 GiB] 3dshapes_gzip4_3-64-3_4096__2181.78eps.h5     | 2181.78 |  36.000 KiB |  36.000 KiB
    # [191.422 MiB|  1.869 GiB] 3dshapes_gzip4_3-64-1_4096__1835.48eps.h5     | 1835.48 |  36.000 KiB |  12.000 KiB
    # hdf5_3dshapes_resave(image_chunks=3, image_block_size=64, image_channels=3, label_chunks=4096, compression='gzip', compression_lvl=4, stop_early=True, dry_run=False, test=True)
    # hdf5_3dshapes_resave(image_chunks=3, image_block_size=64, image_channels=1, label_chunks=4096, compression='gzip', compression_lvl=4, stop_early=True, dry_run=False, test=True)

    # GOOD but cpu usage?
    # [224.992 MiB|  2.197 GiB] 3dshapes_gzip4_1-64-3_4096__2631.94eps.h5     | 2631.94 |  12.000 KiB |  12.000 KiB
    # [212.906 MiB|  2.079 GiB] 3dshapes_gzip4_1-64-1_4096__2165.64eps.h5     | 2165.64 |  12.000 KiB |   4.000 KiB
    # hdf5_3dshapes_resave(image_chunks=1, image_block_size=64, image_channels=3, label_chunks=4096, compression='gzip', compression_lvl=4, stop_early=True, dry_run=False, test=True)
    # hdf5_3dshapes_resave(image_chunks=1, image_block_size=64, image_channels=1, label_chunks=4096, compression='gzip', compression_lvl=4, stop_early=True, dry_run=False, test=True)

    # GOOD but cpu usage?
    # [221.576 MiB|  2.164 GiB] 3dshapes_gzip9_1-64-3_4096__2634.61eps.h5     | 2634.61 |  12.000 KiB |  12.000 KiB
    # [208.458 MiB|  2.036 GiB] 3dshapes_gzip9_1-64-1_4096__2182.76eps.h5     | 2182.76 |  12.000 KiB |   4.000 KiB
    # hdf5_3dshapes_resave(image_chunks=1, image_block_size=64, image_channels=3, label_chunks=4096, compression='gzip', compression_lvl=9, stop_early=True, dry_run=False, test=True)
    # hdf5_3dshapes_resave(image_chunks=1, image_block_size=64, image_channels=1, label_chunks=4096, compression='gzip', compression_lvl=9, stop_early=True, dry_run=False, test=True)

    # BEST but slightly less compression, CPU usage?
    # [306.324 MiB|  2.991 GiB] 3dshapes_lzfNone_1-64-3_4096__2908.88eps.h5   | 2908.88 |  12.000 KiB |  12.000 KiB
    # [257.929 MiB|  2.519 GiB] 3dshapes_lzfNone_1-64-1_4096__2361.23eps.h5   | 2361.23 |  12.000 KiB |   4.000 KiB
    # hdf5_3dshapes_resave(image_chunks=1, image_block_size=64, image_channels=3, label_chunks=4096, compression='lzf', compression_lvl=None, stop_early=True, dry_run=False, test=True)
    # hdf5_3dshapes_resave(image_chunks=1, image_block_size=64, image_channels=1, label_chunks=4096, compression='lzf', compression_lvl=None, stop_early=True, dry_run=False, test=True)

    # TOO BIG
    # [705.884 MiB|  6.893 GiB] 3dshapes_NoneNone_12-64-3_4096__2563.13eps.h5 | 2563.13 | 144.000 KiB | 144.000 KiB
    # [706.508 MiB|  6.899 GiB] 3dshapes_NoneNone_12-64-1_4096__2269.54eps.h5 | 2269.54 | 144.000 KiB |  48.000 KiB
    # [706.818 MiB|  6.903 GiB] 3dshapes_NoneNone_3-64-3_4096__2837.50eps.h5  | 2837.50 |  36.000 KiB |  36.000 KiB
    # [709.308 MiB|  6.927 GiB] 3dshapes_NoneNone_1-64-3_4096__3029.55eps.h5  | 3029.55 |  12.000 KiB |  12.000 KiB
    # [709.308 MiB|  6.927 GiB] 3dshapes_NoneNone_3-64-1_4096__1986.58eps.h5  | 1986.58 |  36.000 KiB |  12.000 KiB

    # hdf5_3dshapes_resave(image_chunks=1, image_block_size=64, image_channels=3, label_chunks=4096, compression=None, compression_lvl=None, stop_early=True, dry_run=False, test=True)
    # hdf5_3dshapes_resave(image_chunks=3, image_block_size=64, image_channels=3, label_chunks=4096, compression=None, compression_lvl=None, stop_early=True, dry_run=False, test=True)
    # hdf5_3dshapes_resave(image_chunks=12, image_block_size=64, image_channels=3, label_chunks=4096, compression=None, compression_lvl=None, stop_early=True, dry_run=False, test=True)
    # hdf5_3dshapes_resave(image_chunks=3, image_block_size=64, image_channels=1, label_chunks=4096, compression=None, compression_lvl=None, stop_early=True, dry_run=False, test=True)
    # hdf5_3dshapes_resave(image_chunks=12, image_block_size=64, image_channels=1, label_chunks=4096, compression=None, compression_lvl=None, stop_early=True, dry_run=False, test=True)

    print_info(get_info_from_filenames())

    class PartialTestDatasetOpenInGetItem(Dataset):
        def __init__(self, file):
            self.file = file
        def __len__(self):
            return 48000
        def __getitem__(self, item):
            with h5py.File(self.file, 'r') as data:
                return data['images'][item]

    class PartialTestDatasetOpenInInit(Dataset):
        def __init__(self, file):
            self.file = file
            self.data = h5py.File(self.file, 'r')['images']
        def __len__(self):
            return 48000
        def __getitem__(self, item):
            return self.data[item]


    test_random_accesses = False
    test_dataloader = True

    files = [
        'data/dataset/3dshapes_gzip4_64-32-1_4096__406.75eps.h5',  # OLD
        'data/dataset/3dshapes_lzfNone_1-64-3_4096__2908.88eps.h5',
        'data/dataset/3dshapes_gzip9_1-64-3_4096__2634.61eps.h5',
        'data/dataset/3dshapes_gzip4_1-64-3_4096__2631.94eps.h5',
    ]

    for file in files:
        print('TESTING:', file)

        if test_random_accesses:
            with h5py.File(file, 'r') as data:
                eps = hdf5_test_entries_per_second(data, 'images')
                print(f'\tEPS={eps:.3f}')

        if test_dataloader:
            # batch_size=64, num_workers=12 | PartialTestDataset
            # TESTING: data/3dshapes_lzfNone_1-64-3_4096__2908.88eps.h5
            # 100%|████████████████████████████████████████| 750/750 [00:04<00:00, 156.19it/s]
            # TESTING: data/3dshapes_gzip9_1-64-3_4096__2634.61eps.h5
            # 100%|████████████████████████████████████████| 750/750 [00:05<00:00, 147.82it/s]
            # TESTING: data/3dshapes_gzip4_1-64-3_4096__2631.94eps.h5
            # 100%|████████████████████████████████████████| 750/750 [00:05<00:00, 148.57it/s]

            # batch_size=64, num_workers=12 | PartialTestDatasetAlt
            # TESTING: data/3dshapes_lzfNone_1-64-3_4096__2908.88eps.h5
            # 100%|████████████████████████████████████████| 750/750 [00:01<00:00, 426.08it/s]
            # TESTING: data/3dshapes_gzip9_1-64-3_4096__2634.61eps.h5
            # 100%|████████████████████████████████████████| 750/750 [00:01<00:00, 379.92it/s]
            # TESTING: data/3dshapes_gzip4_1-64-3_4096__2631.94eps.h5
            # 100%|████████████████████████████████████████| 750/750 [00:01<00:00, 387.58it/s]

            print('Open in getitem', file)
            for item in tqdm(DataLoader(PartialTestDatasetOpenInGetItem(file), batch_size=64, num_workers=12)): pass
            print('Open in __init__', file)
            for item in tqdm(DataLoader(PartialTestDatasetOpenInInit(file), batch_size=64, num_workers=12)): pass

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
