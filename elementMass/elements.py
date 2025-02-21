import re
from decimal import Decimal
from fractions import Fraction
from typing import Dict, List

import pandas as pd

from .oxidation_states import Oxidation_state
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
        chemical notation of an oxide or element

    Returns
    -------
    pandas.Series
        number of cations in each compound
    """

    cations = pd.Series(index=compounds, name="cations", dtype=int)

    for i in cations.index:
        cations.loc[i] = _decompose(i).iloc[0] if _is_oxide(i) else 1

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


def _get_cation_charge(element: str) -> int:

    if _is_oxide(element):
        oxide = _decompose(element)
        return _get_cation_charge_oxide(oxide)

    return _get_element_charge(element)


def _get_cation_charge_oxide(oxide: pd.Series):

    if len(oxide) > 2:
        raise ValueError("oxide contains more than 1 cations")

    n_O = oxide["O"]
    n_cations = oxide.iloc[0]

    charge = int(n_O * 2 / n_cations)

    return charge


def _get_element_charge(element: str):

    try:
        charge = int(re.sub("\D", "", element))
    except ValueError:
        charge = Oxidation_state[element]

    if not (-5 <= charge <= 9):
        raise ValueError(f"Invalid element charge: {charge}")

    return charge


def _is_oxide(compound: str) -> bool:
    """
    returns true if compound contains more than one element and one of them is oxygen. Does not work for 'Os'.
    """

    elements = re.sub(r"[a-z\d]+", "", compound)  # strip lowercase letters and numbers

    return (len(elements) > 1) & ("O" in elements)


def _is_element(compound: str) -> bool:
    """
    returns true if compound is a single element
    """

    elements = re.sub(r"[a-z\d]+", "", compound)  # strip lowercase letters and numbers

    return len(elements) == 1


def cation_names(compounds: List[str]) -> List:
    """
    Get the name of the first cation for each element in compounds

    Parameters
    ----------
    compound    :   list of str
        chemical notation of compounds or elements

    Returns
    -------
    list
        names of cations
    """

    is_oxide = [_is_oxide(c) for c in compounds]
    # is_element = [_is_element(c) for c in compounds]
    element_names = [
        _decompose(c).index[0] if ox else re.sub(r"\d+", "", c)
        for c, ox in zip(compounds, is_oxide)
    ]

    charges = [_get_cation_charge(c) for c in compounds]
    names = [
        (e if Oxidation_state[e] == charge else f"{e}{int(charge)}")
        for e, charge in zip(element_names, charges)
    ]

    return names


def get_oxide_name(ion: str) -> str:

    try:
        charge = float(re.sub(r"\D", "", ion))
    except ValueError:
        charge = Oxidation_state[ion]

    if charge < 0:
        return ion

    ratio = Fraction(Decimal(charge / 2))

    n_cation = int(ratio.denominator) if ratio.denominator > 1 else ""
    n_O = int(ratio.numerator) if ratio.numerator > 1 else ""

    ion_name = re.sub(r"\d+", "", ion)

    return f"{ion_name}{n_cation}O{n_O}"


def get_oxide_names(ions: List[str]) -> List:

    return [get_oxide_name(ion=ion) for ion in ions]
