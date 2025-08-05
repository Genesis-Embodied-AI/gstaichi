/*******************************************************************************
    Copyright (c) The GsTaichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include "gstaichi/common/core.h"

namespace gstaichi {

template <int dim>
class IndexND;

template <int dim>
using TIndex = IndexND<dim>;

template <int dim>
class RegionND;

template <int dim>
using TRegion = RegionND<dim>;

template <int dim, typename T>
class ArrayND;

template <typename T, int dim>
using TArray = ArrayND<dim, T>;

}  // namespace gstaichi
