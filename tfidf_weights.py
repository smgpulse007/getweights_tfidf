'''Rank non-zero terms from one sparse TF-IDF row.'''

from __future__ import annotations

import math
import heapq
import operator
from typing import Any, Sequence


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f'{name} must be a non-bool integer; got bool')
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f'{name} must be a non-bool integer; got {type(value).__name__}'
        ) from exc
    if normalized < 0:
        raise ValueError(f'{name} must be non-negative; got {normalized}')
    return normalized


def _matrix_shape(matrix: Any) -> tuple[int, int]:
    try:
        raw_shape = matrix.shape
    except AttributeError as exc:
        raise TypeError(
            'matrix must provide a 2-D shape and support sparse row slicing'
        ) from exc

    try:
        shape = tuple(raw_shape)
    except TypeError as exc:
        raise TypeError(f'matrix shape must be a 2-item sequence; got {raw_shape!r}') from exc

    if len(shape) != 2:
        raise ValueError(f'matrix must be 2-D; got shape {raw_shape!r}')

    dimensions: list[int] = []
    for axis, dimension in enumerate(shape):
        if isinstance(dimension, bool):
            raise TypeError(f'matrix shape dimension {axis} must be an integer')
        try:
            normalized = operator.index(dimension)
        except TypeError as exc:
            raise TypeError(
                f'matrix shape dimension {axis} must be an integer; got {dimension!r}'
            ) from exc
        if normalized < 0:
            raise ValueError(
                f'matrix shape dimension {axis} must be non-negative; got {normalized}'
            )
        dimensions.append(normalized)
    return dimensions[0], dimensions[1]


def top_terms(
    matrix: Any,
    feature_names: Sequence[str],
    *,
    row_index: int = 0,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    '''Return the highest-weight nonzero terms from one CSR-style row.

    The matrix is accessed only through a one-row sparse slice exposing
    ``indices`` and ``data``.  This supports scipy CSR matrices and csr_array
    objects without importing scipy or densifying the matrix.
    '''
    row_index = _nonnegative_integer(row_index, 'row_index')
    top_n = _nonnegative_integer(top_n, 'top_n')
    if getattr(matrix, 'format', 'csr') != 'csr':
        raise TypeError('matrix must use CSR format; convert it with matrix.tocsr() first')
    n_rows, n_columns = _matrix_shape(matrix)

    if row_index >= n_rows:
        raise IndexError(
            f'row_index {row_index} is out of bounds for matrix with {n_rows} rows'
        )

    try:
        feature_count = len(feature_names)
    except TypeError as exc:
        raise TypeError(
            'feature_names must be a sized sequence in matrix column order'
        ) from exc
    if feature_count != n_columns:
        raise ValueError(
            f'feature_names length ({feature_count}) does not match matrix column count ({n_columns})'
        )

    try:
        row = matrix[row_index : row_index + 1]
    except Exception as exc:
        raise TypeError(
            'matrix must support sparse one-row slicing with matrix[row_index:row_index + 1]'
        ) from exc

    try:
        row_shape = tuple(row.shape)
    except (AttributeError, TypeError):
        row_shape = None
    if row_shape is not None and row_shape != (1, n_columns):
        raise ValueError(
            f'sparse row slice must have shape (1, {n_columns}); got {row_shape!r}'
        )

    try:
        indices = row.indices
        data = row.data
    except AttributeError as exc:
        raise TypeError(
            'matrix[row_index:row_index + 1] must expose sparse .indices and .data; '
            'dense rows are not supported'
        ) from exc

    try:
        index_count = len(indices)
        data_count = len(data)
    except TypeError as exc:
        raise TypeError('sparse row indices and data must be sized sequences') from exc
    if index_count != data_count:
        raise ValueError(
            f'sparse row indices and data lengths differ ({index_count} != {data_count})'
        )

    scores: dict[int, float] = {}
    for position, (raw_column, raw_weight) in enumerate(zip(indices, data)):
        if isinstance(raw_column, bool):
            raise TypeError(
                f'sparse column id at position {position} must be an integer; got bool'
            )
        try:
            column = operator.index(raw_column)
        except TypeError as exc:
            raise TypeError(
                f'sparse column id at position {position} must be an integer; '
                f'got {raw_column!r}'
            ) from exc
        if column < 0 or column >= n_columns:
            raise ValueError(
                f'sparse column id {column} at position {position} is out of range '
                f'for a matrix with {n_columns} columns'
            )

        if isinstance(raw_weight, bool):
            raise TypeError(
                f'sparse weight at position {position} must be a finite non-negative number; '
                'got bool'
            )
        try:
            weight = float(raw_weight)
        except OverflowError as exc:
            raise ValueError(
                f'sparse weight at position {position} must be finite; got {raw_weight!r}'
            ) from exc
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f'sparse weight at position {position} must be a finite non-negative number; '
                f'got {raw_weight!r}'
            ) from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f'sparse weight at position {position} must be finite and non-negative; '
                f'got {raw_weight!r}'
            )

        combined = scores.get(column, 0.0) + weight
        if not math.isfinite(combined) or combined < 0:
            raise ValueError(
                f'combined weight for sparse column {column} is not finite and non-negative'
            )
        scores[column] = combined

    ranked = heapq.nsmallest(
        top_n,
        ((column, score) for column, score in scores.items() if score != 0.0),
        key=lambda item: (-item[1], item[0]),
    )
    return [
        (str(feature_names[column]), score)
        for column, score in ranked
    ]
