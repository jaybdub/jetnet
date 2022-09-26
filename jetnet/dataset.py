# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod

from typing import TypeVar, Generic
from pydantic.generics import GenericModel, BaseModel

T = TypeVar("T")


__all__ = ["Dataset"]


class Dataset(GenericModel, Generic[T]):
    
    def init(self):
        pass

    def build(self):
        clone = self.copy(deep=True)
        clone.init()
        return clone
        
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> T:
        raise NotImplementedError


class _DatasetFilter:
    def __init__(self, dataset, expr):
        self.dataset = dataset
        self.expr = eval(expr)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.expr(self.dataset[index])


class DatasetFilter(BaseModel):

    dataset: BaseModel
    expr: str = "lambda x: x"

    def build(self):
        return _DatasetFilter(self.dataset.build(), self.expr)