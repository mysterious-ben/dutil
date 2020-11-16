import Levenshtein  # noqa: F401
from fuzzywuzzy import fuzz

_translation_table = dict.fromkeys(map(ord, r".,`'!@#$()=^"), None)
_replace_dict = {
    ' incorporated': ' inc',
    ' corporation': ' corp',
    ' company': ' co',
    ' limited': ' ltd',
    ' and': ' &',
    ' corp': ' inc',
    ' ltd': ' inc',
    ' llc': ' inc',
    ' plc': ' inc',
}
_abbr_dict = {
    'hp': 'hewlett-packard',
}


def _translate(x: str):
    x = x.lower().translate(_translation_table)
    for k, v in _replace_dict.items():
        x = x.replace(k, v)
    if x.startswith('the '):
        x = x[4:] + ' the'
    x = x.strip()
    x = _abbr_dict.get(x, x)
    return x


def compare_companies(x: str, y: str) -> float:
    """Compare names of two companies taken into account typical idiosyncrasies in naming
    
    :param x: first company name
    :param y: second company name
    :return: matching rating from 0 (no match) to 1 (a perfect match)
    """
    x = _translate(x)
    y = _translate(y)
    return fuzz.partial_ratio(x, y) / 100
