import logging

log = logging.getLogger(__name__)

# ========================================================================= #
# io                                                                        #
# ========================================================================= #


def ensure_dir_exists(*path):
    import os
    # join path if not a string
    if not isinstance(path, str):
        path = os.path.join(*path)
    # remove file part of directory
    # TODO: this function is useless
    if os.path.isfile(path):
        path = os.path.dirname(path)
    # create missing directory
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        log.info(f'created missing directories: {path}')
    # return directory
    return path


def basename_from_url(url):
    import os
    from urllib.parse import urlparse
    return os.path.basename(urlparse(url).path)


def download_file(url, save_path=None, overwrite_existing=False, chunk_size=4096):
    import requests
    import os
    from tqdm import tqdm

    if save_path is None:
        save_path = basename_from_url(url)
        log.info(f'inferred save_path="{save_path}"')

    # split path
    # TODO: also used in base.py for processing, convert to with syntax.
    path_dir, path_base = os.path.split(save_path)
    ensure_dir_exists(path_dir)

    if not path_base:
        raise Exception('Invalid save path: "{save_path}"')

    # check save path isnt there
    if not overwrite_existing and os.path.isfile(save_path):
        raise Exception(f'File already exists: "{save_path}" set overwrite_existing=True to overwrite.')

    # we download to a temporary file in case there is an error
    temp_download_path = os.path.join(path_dir, f'.{path_base}.download.temp')

    # open the file for saving
    with open(temp_download_path, "wb") as file:
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        # cast to integer if content-length exists on response
        if total_length is not None:
            total_length = int(total_length)

        # download with progress bar
        with tqdm(total=total_length, desc=f'Downloading "{path_base}"', unit='B', unit_scale=True, unit_divisor=1024) as progress:
            for data in response.iter_content(chunk_size=chunk_size):
                file.write(data)
                progress.update(chunk_size)

    # remove if we can overwrite
    if overwrite_existing and os.path.isfile(save_path):
        # TODO: is this necessary?
        os.remove(save_path)

    # rename temp download
    os.rename(temp_download_path, save_path)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
