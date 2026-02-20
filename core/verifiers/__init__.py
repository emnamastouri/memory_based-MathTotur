from core.verifiers.algebra_equations import AlgebraEquationsVerifier
from core.verifiers.calculus import CalculusVerifier
from core.verifiers.sequences import SequencesVerifier
from core.verifiers.linear_algebra import LinearAlgebraVerifier
from core.verifiers.stats import StatsVerifier
from core.verifiers.complex_numbers import ComplexNumbersVerifier

ALL_VERIFIERS = [
    AlgebraEquationsVerifier(),
    CalculusVerifier(),
    SequencesVerifier(),
    LinearAlgebraVerifier(),
    StatsVerifier(),
    ComplexNumbersVerifier(),
]