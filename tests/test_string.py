import pytest

from dutil.string import compare_companies

eps = 0.00001


@pytest.mark.parametrize(
    'name_1, name_2, rating',
    [
        ('Apple Inc.', 'Apple Inc.', 1.),
        ('Apple Inc.', 'xxxxx', 0),
        ('Apple Inc.', 'Apple Incorporated', 1.),
        ('Apple Inc.', 'Apple Limited', 1.),
        ('Apple Inc.', 'Apple', 1.),
        ('Aarons Holdings Company Inc.', "Aaron's, Inc.", .7),
        ('The Tea Company', 'Tea Company (the)', 1.),
        ('HP', 'Hewlett-Packard', 1.),
        ('HP', 'HewlettPackard', .93),
    ]
)
def test_compare_companies(name_1: str, name_2: str, rating: float):
    computed_rating = compare_companies(name_1, name_2)
    assert abs(rating - computed_rating) < eps
