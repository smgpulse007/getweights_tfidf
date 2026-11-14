import math
import unittest

from tfidf_weights import top_terms


class _FakeCSRRow:
    def __init__(self, indices, data, n_columns):
        self.indices = indices
        self.data = data
        self.shape = (1, n_columns)

    def toarray(self):
        raise AssertionError('the sparse row must never be densified')

    def todense(self):
        raise AssertionError('the sparse row must never be densified')

    def __array__(self, *args, **kwargs):
        raise AssertionError('the sparse row must never be densified')


class _FakeCSR:
    def __init__(self, rows, n_columns):
        self._rows = rows
        self.shape = (len(rows), n_columns)

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step is not None:
            raise AssertionError('the matrix must be accessed with a sparse row slice')
        if key.start is None or key.stop != key.start + 1:
            raise AssertionError('expected exactly one requested row')
        if key.start < 0 or key.start >= self.shape[0]:
            raise IndexError(key.start)
        indices, data = self._rows[key.start]
        return _FakeCSRRow(indices, data, self.shape[1])

    def toarray(self):
        raise AssertionError('the matrix must never be densified')

    def todense(self):
        raise AssertionError('the matrix must never be densified')


class _ShapeOnly:
    def __init__(self, shape):
        self.shape = shape


class TopTermsTests(unittest.TestCase):
    def test_zero_rows_are_ignored(self):
        matrix = _FakeCSR([([], []), ([0, 2], [0.0, 0.0])], 3)
        names = ['a', 'b', 'c']
        self.assertEqual(top_terms(matrix, names, row_index=0), [])
        self.assertEqual(top_terms(matrix, names, row_index=1), [])

    def test_duplicate_columns_are_combined_and_zero_scores_removed(self):
        matrix = _FakeCSR(
            [([2, 1, 1, 0], [0.0, 0.25, 0.25, 0.5])],
            3,
        )
        self.assertEqual(
            top_terms(matrix, ['zero', 'one', 'two']),
            [('zero', 0.5), ('one', 0.5)],
        )

    def test_ties_use_ascending_column_order(self):
        matrix = _FakeCSR([([2, 0, 1], [0.5, 0.5, 0.2])], 3)
        self.assertEqual(
            top_terms(matrix, ['first', 'second', 'third'], top_n=2),
            [('first', 0.5), ('third', 0.5)],
        )

    def test_top_n_is_exact_when_enough_nonzero_terms_exist(self):
        matrix = _FakeCSR([([0, 1, 2, 3], [0.1, 0.4, 0.3, 0.0])], 4)
        names = ['zero', 'one', 'two', 'three']
        self.assertEqual(
            top_terms(matrix, names, top_n=2),
            [('one', 0.4), ('two', 0.3)],
        )
        self.assertEqual(top_terms(matrix, names, top_n=0), [])
        self.assertEqual(
            top_terms(matrix, names, top_n=20),
            [('one', 0.4), ('two', 0.3), ('zero', 0.1)],
        )

    def test_input_is_not_mutated(self):
        indices = [1, 1, 0]
        data = [0.2, 0.3, 0.4]
        names = ['zero', 'one']
        matrix = _FakeCSR([(indices, data)], 2)
        before_indices = indices.copy()
        before_data = data.copy()
        before_names = names.copy()

        top_terms(matrix, names)

        self.assertEqual(indices, before_indices)
        self.assertEqual(data, before_data)
        self.assertEqual(names, before_names)

    def test_invalid_dimensions_and_feature_count(self):
        with self.assertRaisesRegex(ValueError, '2-D'):
            top_terms(_ShapeOnly((3,)), ['a', 'b', 'c'])
        with self.assertRaisesRegex(ValueError, '2-D'):
            top_terms(_ShapeOnly((1, 2, 3)), ['a', 'b'])

        matrix = _FakeCSR([([], [])], 2)
        with self.assertRaisesRegex(ValueError, 'feature_names length'):
            top_terms(matrix, ['only-one'])

    def test_invalid_row_arguments_and_bounds(self):
        matrix = _FakeCSR([([], []), ([], [])], 0)
        for value in (True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    top_terms(matrix, [], row_index=value)
        with self.assertRaises(ValueError):
            top_terms(matrix, [], row_index=-1)
        with self.assertRaisesRegex(IndexError, 'out of bounds'):
            top_terms(matrix, [], row_index=2)

        for value in (False, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    top_terms(matrix, [], top_n=value)
        with self.assertRaises(ValueError):
            top_terms(matrix, [], top_n=-1)

    def test_invalid_weights(self):
        for weight in (math.nan, math.inf, -math.inf, -0.1):
            with self.subTest(weight=weight):
                matrix = _FakeCSR([([0], [weight])], 1)
                with self.assertRaisesRegex(ValueError, 'finite'):
                    top_terms(matrix, ['term'])

    def test_out_of_range_sparse_column_ids(self):
        for column in (-1, 2):
            with self.subTest(column=column):
                matrix = _FakeCSR([([column], [1.0])], 2)
                with self.assertRaisesRegex(ValueError, 'out of range'):
                    top_terms(matrix, ['a', 'b'])

    def test_non_csr_sparse_layout_is_rejected(self):
        matrix = _FakeCSR([([0], [1.0])], 1)
        matrix.format = 'csc'
        with self.assertRaisesRegex(TypeError, 'CSR format'):
            top_terms(matrix, ['a'])

    def test_duplicate_weight_overflow_is_rejected(self):
        matrix = _FakeCSR([([0, 0], [1e308, 1e308])], 1)
        with self.assertRaisesRegex(ValueError, 'combined weight'):
            top_terms(matrix, ['a'])

    def test_mismatched_sparse_arrays_are_rejected(self):
        matrix = _FakeCSR([([0, 1], [0.5])], 2)
        with self.assertRaisesRegex(ValueError, 'lengths differ'):
            top_terms(matrix, ['a', 'b'])


try:
    import scipy.sparse as _scipy_sparse
    from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
except ImportError:
    _HAVE_OPTIONAL = False
else:
    _HAVE_OPTIONAL = True


class OptionalIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        _HAVE_OPTIONAL,
        'scipy and scikit-learn are optional integration-test dependencies',
    )
    def test_real_sklearn_csr_matrix_and_csr_array(self):
        vectorizer = _TfidfVectorizer()
        matrix = vectorizer.fit_transform(['red fox', 'blue fox'])
        names = vectorizer.get_feature_names_out()
        matrices = [matrix]
        if hasattr(_scipy_sparse, 'csr_array'):
            matrices.append(_scipy_sparse.csr_array(matrix))

        for candidate in matrices:
            with self.subTest(matrix_type=type(candidate).__name__):
                result = top_terms(candidate, names, row_index=0, top_n=2)
                self.assertEqual([name for name, _ in result], ['red', 'fox'])
                self.assertTrue(all(math.isfinite(weight) for _, weight in result))
        with self.assertRaisesRegex(TypeError, 'CSR format'):
            top_terms(matrix.tocsc(), names)


if __name__ == '__main__':
    unittest.main()
