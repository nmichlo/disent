#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

import base64
import dataclasses
import inspect
import io
import os
from typing import Optional
from typing import Union

import torch

from disent.util.inout.paths import ensure_parent_dir_exists


# ========================================================================= #
# Github Upload Utility Functions                                           #
# ========================================================================= #


def gh_get_repo(repo: str = None):
    from github import Github
    # get token str
    token = os.environ.get('GITHUB_TOKEN', '')
    if not token.strip():
        raise ValueError('`GITHUB_TOKEN` env variable has not been set!')
    assert isinstance(token, str)
    # get repo str
    if repo is None:
        repo = os.environ.get('GITHUB_REPO', '')
        if not repo.strip():
            raise ValueError('`GITHUB_REPO` env variable has not been set!')
    assert isinstance(repo, str)
    # get repo
    return Github(token).get_repo(repo)


def gh_get_branch(repo: 'Repository', branch: str = None, source_branch: str = None, allow_new_branch: bool = True) -> 'Branch':
    from github import GithubException
    # check branch
    assert isinstance(branch, str) or (branch is None)
    assert isinstance(source_branch, str) or (source_branch is None)
    # get default branch
    if branch is None:
        branch = repo.default_branch
    # retrieve branch
    try:
        return repo.get_branch(branch)
    except GithubException as e:
        if not allow_new_branch:
            raise RuntimeError(f'Creating branch disabled, set `allow_new_branch=True`: {repr(branch)}')
        print(f'Creating missing branch: {repr(branch)}')
        sb = repo.get_branch(repo.default_branch if (source_branch is None) else source_branch)
        repo.create_git_ref(ref='refs/heads/' + branch, sha=sb.commit.sha)
        return repo.get_branch(branch)


@dataclasses.dataclass
class WriteResult:
    commit: 'Commit'
    content: 'ContentFile'


def gh_write_file(repo: 'Repository', path: str, content: Union[str, bytes], branch: str = None, allow_new_file=True, allow_overwrite_file=False, allow_new_branch=True) -> WriteResult:
    from github import UnknownObjectException
    # get branch
    branch = gh_get_branch(repo, branch, allow_new_branch=allow_new_branch).name
    # check that the file exists
    try:
        sha = repo.get_contents(path, ref=branch).sha
    except UnknownObjectException:
        sha = None
    # handle file exists or not
    if sha is None:
        if not allow_new_file:
            raise RuntimeError(f'Creating file disabled, set `allow_new_file=True`: {repr(path)}')
        result = repo.create_file(path=path, message=f'Created File: {path}', content=content, branch=branch)
    else:
        if not allow_overwrite_file:
            raise RuntimeError(f'Overwriting file disabled, `set allow_overwrite_file=True`: {repr(path)}')
        result = repo.update_file(path=path, message=f'Updated File: {path}', content=content, branch=branch, sha=sha)
    # result is a dict: {'commit': github.Commit, 'content': github.ContentFile}
    return WriteResult(**result)


# ========================================================================= #
# Github Upload Utility Class                                               #
# ========================================================================= #


class GithubWriter(object):

    def __init__(self, repo: str = None, branch: str = None, allow_new_file=True, allow_overwrite_file=True,  allow_new_branch=True):
        self._kwargs = dict(
            repo=gh_get_repo(repo=repo),
            branch=branch,
            allow_new_file=allow_new_file,
            allow_overwrite_file=allow_overwrite_file,
            allow_new_branch=allow_new_branch,
        )

    def write_file(self, path: str, content: Union[str, bytes]):
        return gh_write_file(
            path=path,
            content=content,
            **self._kwargs,
        )


# ========================================================================= #
# Torch Save Utils                                                          #
# ========================================================================= #


def torch_save_bytes(model) -> bytes:
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    return buffer.read()


def torch_save_base64(model) -> str:
    b = torch_save_bytes(model)
    return base64.b64encode(b).decode('ascii')


def torch_load_bytes(b: bytes):
    return torch.load(io.BytesIO(b))


def torch_load_base64(s: str):
    b = base64.b64decode(s.encode('ascii'))
    return torch_load_bytes(b)


# ========================================================================= #
# write                                                                     #
# ========================================================================= #


def _split_special_path(path):
    if path.startswith('github:'):
        # get github repo and path
        path = path[len('github:'):]
        repo, path = os.path.join(*path.split('/')[:2]), os.path.join(*path.split('/')[2:])
        # check paths
        assert repo.strip() and len(repo.split('/')) == 2
        assert path.strip() and len(repo.split('/')) >= 1
        # return components
        return 'github', (repo, path)
    else:
        return 'local', path


def torch_write(path: str, model):
    path_type, path = _split_special_path(path)
    # handle cases
    if path_type == 'github':
        path, repo = path
        # get the name of the path
        ghw = GithubWriter(repo)
        ghw.write_file(path=path, content=torch_save_bytes(model))
        print(f'Saved in repo: {repr(path)} to file: {repr(repo)}')
    elif path_type == 'local':
        torch.save(model, ensure_parent_dir_exists(path))
        print(f'Saved to file: {repr(path)}')
    else:
        raise KeyError(f'unknown path type: {repr(path_type)}')


# ========================================================================= #
# Files                                                                     #
# ========================================================================= #


def _make_rel_path(*path_segments, is_file=True, _calldepth=0):
    assert not os.path.isabs(os.path.join(*path_segments)), 'path must be relative'
    # get source
    stack = inspect.stack()
    module = inspect.getmodule(stack[_calldepth+1].frame)
    reldir = os.path.dirname(module.__file__)
    # make everything
    path = os.path.join(reldir, *path_segments)
    folder_path = os.path.dirname(path) if is_file else path
    os.makedirs(folder_path, exist_ok=True)
    return path


def _make_rel_path_add_ext(*path_segments, ext='.png', _calldepth=0):
    # make path
    path = _make_rel_path(*path_segments, is_file=True, _calldepth=_calldepth+1)
    if not os.path.splitext(path)[1]:
        path = f'{path}{ext}'
    return path


def make_rel_path(*path_segments, is_file=True):
    return _make_rel_path(*path_segments, is_file=is_file, _calldepth=1)


def make_rel_path_add_ext(*path_segments, ext='.png'):
    return _make_rel_path_add_ext(*path_segments, ext=ext, _calldepth=1)


def plt_rel_path_savefig(rel_path: Optional[str], save: bool = True, show: bool = True, ext='.png', dpi: Optional[int] = None, **kwargs):
    import matplotlib.pyplot as plt
    if save and (rel_path is not None):
        path = _make_rel_path_add_ext(rel_path, ext=ext, _calldepth=2)
        plt.savefig(path, dpi=dpi, **kwargs)
        print(f'saved: {repr(path)}')
    if show:
        plt.show(**kwargs)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
