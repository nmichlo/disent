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

import dataclasses
import os
from typing import Union

from git import Commit
from github import ContentFile
from github import Github
from github import GithubException
from github import UnknownObjectException
from github.Branch import Branch
from github.Repository import Repository


# ========================================================================= #
# Github Upload Utility Functions                                           #
# ========================================================================= #


def gh_get_repo(repo: str = None):
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


def gh_get_branch(repo: Repository, branch: str = None, source_branch: str = None, allow_new_branch: bool = True) -> Branch:
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
    commit: Commit
    content: ContentFile


def gh_write_file(repo: Repository, path: str, content: Union[str, bytes], branch: str = None, allow_new_file=True, allow_overwrite_file=False, allow_new_branch=True) -> WriteResult:
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
