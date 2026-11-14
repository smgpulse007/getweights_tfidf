from tfidf_weights import top_terms


def main() -> None:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise SystemExit(
            'The optional example requires scipy and scikit-learn; '
            'install them with: python -m pip install scipy scikit-learn'
        ) from exc

    documents = [
        'red fox jumps',
        'blue fox sleeps',
    ]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    print('Top terms from two synthetic documents (sparse CSR input)')
    for row_index, document in enumerate(documents):
        print(f'\nDocument {row_index}: {document}')
        for term, weight in top_terms(matrix, feature_names, row_index=row_index, top_n=3):
            print(f'  {term:<10} {weight:.4f}')


if __name__ == '__main__':
    main()
