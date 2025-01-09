# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# New Supported Instructions:
# BUILD_LIST (new)
# BINARY_SUBSCR
# DELETE_SUBSCR

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


class ListIterable:
    def __init__(self):
        self._list = [1, 2, 3]

    def __iter__(self):
        return iter(self._list)


class ListIterableMagicMethod:
    def __init__(self):
        self._list = [1, 2, 3]

    def __iter__(self):
        return self._list.__iter__()


def list_iterable(x: paddle.Tensor):
    iterable_1 = ListIterable()
    iterable_2 = ListIterableMagicMethod()
    for i in iterable_1:
        x += i
    for j in iterable_2:
        x += j
    return x


class TupleIterable:
    def __init__(self):
        self._tuple = (1, 2, 3)

    def __iter__(self):
        return iter(self._tuple)


class TupleIterableMagicMethod:
    def __init__(self):
        self._tuple = (1, 2, 3)

    def __iter__(self):
        return self._tuple.__iter__()


def tuple_iterable(x: paddle.Tensor):
    iterable_1 = TupleIterable()
    iterable_2 = TupleIterableMagicMethod()
    for i in iterable_1:
        x += i
    for j in iterable_2:
        x += j
    return x


class DictIterable:
    def __init__(self):
        self._dict = {0: 1, 1: 2, 2: 3}

    def __iter__(self):
        return iter(self._dict)


class DictIterableMagicMethod:
    def __init__(self):
        self._dict = {0: 1, 1: 2, 2: 3}

    def __iter__(self):
        return self._dict.__iter__()


def dict_iterable(x: paddle.Tensor):
    iterable_1 = DictIterable()
    iterable_2 = DictIterableMagicMethod()
    for i in iterable_1:
        x += i
    for j in iterable_2:
        x += j
    return x


class RangeIterable:
    def __init__(self):
        pass

    def __iter__(self):
        return range(5).__iter__()


class RangeIterableMagicMethod:
    def __init__(self):
        pass

    def __iter__(self):
        return range(5).__iter__()


def range_iterable(x: paddle.Tensor):
    iterable_1 = RangeIterable()
    iterable_2 = RangeIterableMagicMethod()
    for i in iterable_1:
        x += i
    for j in iterable_2:
        x += j
    return x


class TestIterable(TestCaseBase):
    def test_list_iterable(self):
        self.assert_results(list_iterable, paddle.to_tensor(0))

    def test_tuple_iterable(self):
        self.assert_results(tuple_iterable, paddle.to_tensor(0))

    def test_dict_iterable(self):
        self.assert_results(dict_iterable, paddle.to_tensor(0))

    def test_range_iterable(self):
        self.assert_results(range_iterable, paddle.to_tensor(0))


if __name__ == "__main__":
    unittest.main()
