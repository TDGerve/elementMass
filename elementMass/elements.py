import re
from typing import List

import pandas as pd

from .Periodic_table import get_periodic_table

periodic_table = get_periodic_table()


def _find_elements(compound: str):
    elements = re.findall("([A-Z][^A-Z]*)", str(compound))

    # Raise an error if no elements are found
    if len(elements) < 1:
        raise ValueError(f"'{compound}' does not contain valid elements")

    for element in elements:
        # Raise an error for invalid elements with more than 1 lower case character
        if sum(c.islower() for c in element) > 1:
            raise ValueError(f"'{element}' is not a valid element")
        # Check for non-word characters
        elif len(re.findall(r"\w|\.", element)) != len(element):
            raise ValueError(f"'{element}' contains an invalid character")

    # Raise an error if there are any leftover characters
    length_elements = sum(len(s) for s in elements)
    if len(compound) != length_elements:
        raise ValueError(
            f"There are leftover characters in '{compound}'; elements found: {elements}"
        )

    return elements


def _find_quantity(element: str):
    element_quantity = re.findall(r"(\D+|\d[^A-Z]*)", element)

    if len(element_quantity) < 2:
        element_quantity.append(1)

    return element_quantity


def _decompose(compound: str):
    elements = [_find_quantity(i) for i in _find_elements(compound)]

    elements_pd = pd.DataFrame(elements, columns=["element", "quantity"]).set_index(
        "element"
    )

    return elements_pd.astype(float).squeeze("columns")


def calculate_weight(compound: str) -> float:
    """
    Get the atomic mass of a compound or element

    Parameters
    ----------
    compound    :   str
        chemical notation of a compound or element

    Returns
    -------
    float
        weight in atomic mass units
    """

    elements = _decompose(compound)

    return (periodic_table[elements.index] * elements).sum()


def compound_weights(compounds: List[str]) -> pd.Series:
    """
    Get the atomic mass of a compounds or elements

    Parameters
    ----------
    compound    :   list of str
        chemical notation of compounds or elements

    Returns
    -------
    pandas.Series
        weights in atomic mass units
    """

    weights = pd.Series(index=compounds, name="weights", dtype="float64")

    for i in weights.index:
        weights[i] = calculate_weight(i)

    return weights


def cation_numbers(compounds: List[str]) -> pd.Series:
    """
    Get the cation amount in compounds

    Parameters
    ----------
    compound    :   list of str
        chemical notation of compounds or elements

    Returns
    -------
    pandas.Series
        number of cations in each compound
    """

    cations = pd.Series(index=compounds, name="cations", dtype=int)

    for i in cations.index:
        cations.loc[i] = _decompose(i)[0]

    return cations


def oxygen_numbers(compounds: List[str]) -> pd.Series:
    """
    Get the oxygen amount in compounds

    Parameters
    ----------
    compound    :   list of str
        chemical notation of compounds or elements

    Returns
    -------
    pandas.Series
        number of oxygen in each compound
    """

    oxygen = pd.Series(index=compounds, name="oxygen", dtype=int)

    for i in oxygen.index:
        try:
            oxygen[i] = _decompose(i)["O"]
        except KeyError:
            oxygen[i] = 0

    return oxygen


def cation_names(compounds: List[str]) -> List:
    """
    Get the name of the first cation in a compound

    Parameters
    ----------
    compound    :   list of str
        chemical notation of compounds or elements

    Returns
    -------
    list
        names of cations
    """

    names = [_decompose(oxide).index[0] for oxide in compounds]

    if "Fe2O3" in compounds:
        idx = compounds.index("Fe2O3")
        names[idx] = "Fe3"

    return names
