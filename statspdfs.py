# Implémentation des pdfs utilisées

import scipy.stats as stats
from scipy.integrate import quad


def standard_normal(x):
    """Calcule la pdf d'une loi normale centrée réduite."""
    return stats.norm().pdf(x)


def chi2(x, df):
    """Calcule la pdf d'une loi du chi2 à df degrés de liberté."""
    return stats.chi2(df).pdf(x)


def student(x, df):
    """Calcule la pdf d'une loi de Student à df degrés de liberté."""
    return stats.t(df).pdf(x)


def fisher(x, d1, d2):
    return stats.f(d1, d2).pdf(x)