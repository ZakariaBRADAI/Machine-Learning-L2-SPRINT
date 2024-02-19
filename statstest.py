import numpy as np
from scipy.integrate import quad
import scipy.stats as stats
import statspdfs


class InvalidAlternateHypothesis(Exception):
    """Gère les exceptions liées aux mauvais types d'encadrement."""

    def __init__(self, h1, base_message="Type de test invalide", *args, **kwargs):
        """Initialise le message"""
        msg = (
            f"{base_message} : {h1} \n"
            "Valeurs possibles du paramètre 'H1' : 'two-sided' - 'greater' - 'less' \n"
            "Veuillez entrer une valeur valide"
        )
        super().__init__(msg, *args, **kwargs)


class InvalidAlphaValue(Exception):
    """Gère les exceptions liées aux valeurs incorrectes du risque alpha."""

    def __init__(
        self, alpha, base_message="Valeur incorrecte du risque alpha", *args, **kwargs
    ):
        """Initialise le message"""
        msg = (
            f"{base_message} : {alpha} \n"
            "Valeurs possibles du paramètre 'alpha' : [0;1[ \n"
        )
        super().__init__(msg, *args, **kwargs)

class InvalidProportionValue(Exception):
    """Gère les exceptions liées aux valeurs incorrectes du risque alpha."""

    def __init__(
        self, p0, base_message="Valeur incorrecte de p0", *args, **kwargs
    ):
        """Initialise le message"""
        msg = (
            f"{base_message} : {p0} \n"
            "Valeurs possibles d'une proportion : [0;1] \n"
        )
        super().__init__(msg, *args, **kwargs)

def check_alpha(alpha):
    '''Vérifie si l'on a entré une valeur correcte.'''
    if (alpha < 0) or (alpha >= 1):
        raise InvalidAlphaValue(alpha)


def check_proportion(p0):
    '''Vérifie si l'on a entré une valeur correcte.'''
    if (p0 < 0) or (p0 > 1):
        raise InvalidProportionValue(p0)

def check_h1(h1):
    """Vérifie si l'on a entré une hypothèse alternative correcte."""

    if h1 not in ["two-sided", "greater", "less"]:
        raise InvalidAlternateHypothesis(h1)


def compute_pvalue(pdf, sup, inf, args_pdf=()):
    """Calcule l'intégrale donnant la p value"""
    return quad(pdf, sup, inf, args=args_pdf)[0]


def conclude_pvalue(p_value, alpha):
    """Conclut les tests statistiques."""
    if p_value <= alpha:
        print(f"H0 est rejetée avec un risque de {alpha * 100}% \n")
    else:
        print(f"H0 est n'est pas rejetée avec un risque de {alpha * 100}% \n")


def z_test(z, h1):
    """
    Calcule la p-value d'un test en Z.
    """
    match h1:
        case "two-sided":
            p_value = 2 * compute_pvalue(statspdfs.standard_normal, np.abs(z), np.inf)
        case "greater":
            p_value = compute_pvalue(statspdfs.standard_normal, -np.inf, z)
        case "less":
            p_value = compute_pvalue(statspdfs.standard_normal, z, np.inf)
    print(f"Z = {z}, p value = {p_value}")
    return p_value


def t_test(t, h1, df):
    """
    Calcule la p-value d'un test en T.
    """
    match h1:
        case "two-sided":
            p_value = 2 * compute_pvalue(
                statspdfs.student, np.abs(t), np.inf, args_pdf=df
            )
        case "greater":
            p_value = compute_pvalue(statspdfs.student, -np.inf, t, args_pdf=df)
        case "less":
            p_value = compute_pvalue(statspdfs.student, t, np.inf, args_pdf=df)
    print(f"T = {t}, p value = {p_value}")
    return p_value


def chi_test(chi2, h1, df):
    """
    Calcule la p-value d'un test en Chi2.
    """
    match h1:
        case "two-sided":
            i_sup = compute_pvalue(statspdfs.chi2, 0, chi2, args_pdf=df)
            i_inf = compute_pvalue(statspdfs.chi2, chi2, np.inf, args_pdf=df)
            p_value = 2 * min(i_inf, i_sup)
        case "greater":
            p_value = compute_pvalue(statspdfs.chi2, 0, chi2, args_pdf=df)
        case "less":
            p_value = compute_pvalue(statspdfs.chi2, chi2, np.inf, args_pdf=df)
    print(f"K = {chi2}, p value = {p_value}")
    return p_value


#---------Decorators-------------------

def decorator_choose_h1(function):
    '''Fonction décoratrice pour un test dont nous
    choisissons l'hypothèse alternative H1.'''

    def wrapper(*args, **kwargs):

        alpha = args[2]
        h1 = args[3]
        check_alpha(alpha)
        check_h1(h1)
        p_value = function(*args, **kwargs)
        conclude_pvalue(p_value, alpha)
        return p_value

    wrapper.__doc__ = function.__doc__
    return wrapper


def decorator_defined_h1(function):
    '''Fonction décoratrice pour un test dont
    l'hypothèse alternative H1 est prédéfinie.'''

    def wrapper(*args, **kwargs):

        alpha = args[2]
        check_alpha(alpha)
        p_value = function(*args, **kwargs)
        conclude_pvalue(p_value, alpha)
        return p_value

    wrapper.__doc__ = function.__doc__
    return wrapper


# ----------------- TESTS --------------


@decorator_choose_h1
def test_proportion(data, p0, alpha=0.05, h1="two-sided"):
    """
    Teste une proportion par p value
    avec un risque alpha par test en Z.

    Parameters :
    ------------
    data : nd array, échantillon statistique binaire
    p0 : float, proportion théorique des 1
    alpha : float, risque du test, par défaut = 0.05
    h1 : str, hypothèse alternative, par défaut = "two-sided"

    Raises:
    InvalidProportionValue : si p0 n'est pas entre 0 et 1
    InvalidAlphaValue : si alpha n'est pas entre 0 et 1
    InvalidAlternateHypothesis : si h1 n'a pas une valeur correcte ; "two-sided" - "left" - "right"
    """
    check_proportion(p0)
    n = len(data)
    z = np.sqrt(n) * ((np.mean(data) - p0) / np.sqrt(p0 * (1 - p0)))
    p_value = z_test(z, h1)
    return p_value


@decorator_choose_h1
def test_proportion_2samples(data1, data2, alpha=0.05, h1="two-sided"):
    """
    Teste l'égalité de 2 proportions avec un risque alpha.

    Parameters :
    ------------
    data1 : nd array, échantillon 1
    data2 : nd array, échantillon 2
    alpha : float, risque du test, par défaut = 0.05
    h1 : str, hypothèse alternative, par défaut = "two-sided"

    Raises:
    InvalidAlphaValue : si alpha n'est pas entre 0 et 1
    InvalidAlternateHypothesis : si h1 n'a pas une valeur correcte ; "two-sided" - "left" - "right"
    """
    n1 = len(data1)
    n2 = len(data2)
    prop1 = np.mean(data1)
    prop2 = np.mean(data2)
    hat_p = (prop1 + prop2) / (n1 + n2)
    z = (prop1 - prop2) / np.sqrt(hat_p * (1 - hat_p) * ((n1 + n2) / (n1 * n2)))
    p_value = z_test(z, h1)
    print(f"Z = {z}, p value = {p_value}")
    return p_value


@decorator_choose_h1
def test_moyenne(data, mu0, alpha=0.05, h1="two-sided"):
    """
    Teste une moyenne par p value
    avec un risque alpha par test en T.

    Parameters :
    ------------
    data1 : nd array, échantillon
    mu0 : float, moyenne théorique
    alpha : float, risque du test, par défaut = 0.05
    h1 : str, hypothèse alternative, par défaut = "two-sided"

    Raises:
    InvalidAlphaValue : si alpha n'est pas entre 0 et 1
    InvalidAlternateHypothesis : si h1 n'a pas une valeur correcte ; "two-sided" - "left" - "right"
    """
    n = len(data)
    s_prime = np.var(data, ddof=1)
    t = np.sqrt(n) * ((np.mean(data) - mu0) / s_prime)
    p_value = t_test(t, h1, df=n - 1)
    return p_value


@decorator_choose_h1
def test_variance(data, var0, alpha=0.05, h1="two-sided"):
    """
    Teste une moyenne par p value
    avec un risque alpha par test en Chi2.

    Parameters :
    ------------
    data : nd array, échantillon
    var0 : float, variance théorique
    alpha : float, risque du test, par défaut = 0.05
    h1 : str, hypothèse alternative, par défaut = "two-sided"

    Raises:
    InvalidAlphaValue : si alpha n'est pas entre 0 et 1
    InvalidAlternateHypothesis : si h1 n'a pas une valeur correcte ; "two-sided" - "left" - "right"
    """
    n = len(data)
    s_prime2 = np.var(data, ddof=1)
    k = (n - 1) * (s_prime2 / var0)
    p_value = chi_test(k, h1, df=n - 1)
    return p_value

@decorator_defined_h1
def test_egalite_variances_gauss_iid(data1, data2, alpha=0.05):
    """
    Test l'égalité des variances de 2 échantillons gaussiens iid.
    """
    n1 = len(data1)
    n2 = len(data2)
    s_prime2_1 = np.var(data1, ddof=1)
    s_prime2_2 = np.var(data2, ddof=1)
    f = s_prime2_1 / s_prime2_2
    i_inf_fisher = compute_pvalue(statspdfs.fisher, 0, f, args_pdf=(n1, n2))
    i_sup_fisher = compute_pvalue(statspdfs.fisher, f, np.inf, args_pdf=(n1, n2))
    p_value = 2 * min(i_inf_fisher, i_sup_fisher)
    print("Test d'égalité des variances :")
    print(f"F = {f}, p value = {p_value}")
    return p_value

@decorator_defined_h1
def test_egalite_moyennes_gauss_iid(data1, data2, alpha=0.05):
    """
    Teste l'égalité des moyennes de 2 échantillons gaussiens iid
    """
    n1 = len(data1)
    n2 = len(data2)
    mu1 = np.mean(data1)
    mu2 = np.mean(data2)
    s_prime2_1 = np.var(data1, ddof=1)
    s_prime2_2 = np.var(data2, ddof=1)
    numerateur = np.sqrt(n1 + n2 - 2) * (mu1 + mu2)
    denominateur = np.sqrt((n1 + n2) / (n1 * n2)) * np.sqrt(
        (n1 - 1) * s_prime2_1 + (n2 - 1) * s_prime2_2
    )
    t = numerateur / denominateur
    p_value = compute_pvalue(statspdfs.student, np.abs(t), np.inf, args_pdf=n1 + n2 - 2)
    print("Test d'égalité des moyennes :")
    print(f"T = {t}, p value = {p_value}")
    return p_value


def test_egalite_gauss_iid(data1, data2, alpha_var=0.05, alpha_moy=0.05):
    """Teste l'églité de 2 échantillons gaussiens iid
    en faisant un test d'égalité des variances puis
    un test d'égalité des moyennes.

    Parameters :
    ------------
    data1 : nd array, échantillon 1
    data2 : nd array, échantillon 2
    alpha_var : float, risque du test d'égalité des variances, par défaut = 0.05
    alpha_moy : float, risque du test d'égalité des moyennes, par défaut = 0.05

    Raises:
    InvalidAlphaValue : si alpha n'est pas entre 0 et 1
    InvalidAlternateHypothesis : si h1 n'a pas une valeur correcte ; "two-sided" - "left" - "right"
    """
    success_test_moy = False
    p_value_var = test_egalite_variances_gauss_iid(data1, data2, alpha_var)
    success_test_var = p_value_var > alpha_var
    if success_test_var:
        p_value_moy = test_egalite_moyennes_gauss_iid(data1, data2)
        success_test_moy = p_value_moy > alpha_moy
    if success_test_var and success_test_moy:
        print(
            f"Les échantillons sont égaux avec un risque de {alpha_var * 100}% sur l'égalité des variances"
            f"et de {alpha_moy * 100} sur l'égalité des moyennes."
        )
        return p_value_var, p_value_moy
    else:
        print(f"Les échantillons ne sont pas égaux.")
        return 1, 1


@decorator_choose_h1
def test_welch_gauss_iid(data1, data2, alpha=0.05, h1="two-sided"):
    """
    Réalise un test de Welch sur 2 échantillons gaussiens iid.

    Parameters :
    ------------
    data1 : nd array, échantillon 1
    data2 : nd array, échantillon 2
    alpha : float, risque du test, par défaut = 0.05
    h1 : str, hypothèse alternative, par défaut = "two-sided"

    Raises:
    InvalidAlphaValue : si alpha n'est pas entre 0 et 1
    InvalidAlternateHypothesis : si h1 n'a pas une valeur correcte ; "two-sided" - "left" - "right"
    """
    n1 = len(data1)
    n2 = len(data2)
    mu1 = np.mean(data1)
    mu2 = np.mean(data2)
    s_prime2_1 = np.var(data1, ddof=1)
    s_prime2_2 = np.var(data2, ddof=1)
    t = (mu1 - mu2) / np.sqrt(s_prime2_1 / n1 + s_prime2_2 / n2)
    numerateur_nu = (s_prime2_1 / n1 + s_prime2_2 / n2) ** 2
    denominateur_nu = (s_prime2_1**2) / ((n1**2) * (1 - n1)) + (s_prime2_2**2) / (
        (n2**2) * (1 - n2)
    )
    nu = numerateur_nu / denominateur_nu
    p_value = t_test(t, h1, df=nu)
    return p_value


@decorator_choose_h1
def test_ks_th(data, cdf_th, alpha=0.05, h1="two-sided", arguments_cdf=()):
    """
    Test d'adéquation de Komlogorov-Smirnov sur 1 échantillon avec un risque alpha.


    Parameters :
    ------------
    data : nd array, échantillon statistique
    cdf_th : function, fonction de répartition théorique
    alpha : float, risque du test, par défaut = 0.05
    h1 : str, hypothèse alternative, par défaut = "two-sided"
    arguments_cdf : tuple, arguments de la cdf, par défaut = ()

    Raises:
    InvalidAlphaValue : si alpha n'est pas entre 0 et 1
    InvalidAlternateHypothesis : si h1 n'a pas une valeur correcte ; "two-sided" - "left" - "right"
    """
    statistique, p_value = stats.ks_1samp(
        data, cdf_th, args=arguments_cdf, alternative=h1, method="auto"
    )
    print(f"Dn = {statistique}, p value = {p_value}")
    return p_value


@decorator_choose_h1
def test_ks_2samples(data1, data2, alpha=0.05, h1="two-sided"):
    """
    Test d'adéquation de Komlogorov-Smirnov sur 2 échantillons avec un risque alpha.

    Parameters :
    ------------
    data1 : nd array, échantillon 1
    data2 : nd array, échantillon 2
    alpha : float, risque du test, par défaut = 0.05
    h1 : str, hypothèse alternative, par défaut = "two-sided"

    Raises:
    InvalidAlphaValue : si alpha n'est pas entre 0 et 1
    InvalidAlternateHypothesis : si h1 n'a pas une valeur correcte ; "two-sided" - "left" - "right"
    """
    statistique, p_value, statistic_location, statistic_sign = stats.ks_2samp(
        data1, data2, alternative=h1, method="auto"
    )
    print(f"Dn = {statistique}, p value = {p_value}")
    print(
        f"statistic_location : {statistic_location} - statistic_sign : {statistic_sign}"
    )
    return p_value


@decorator_defined_h1
def test_chi2_discret(data, p0, alpha=0.05):
    """
    Test du chi2 dans un cas discret avec un risque alpha.


    Parameters :
    ------------
    data : nd array, échantillon binaire
    p0 : float, proportion théorique des 1
    alpha : float, risque du test, par défaut = 0.05

    Raises:
    InvalidProportionValue : si p0 n'est pas entre 0 et 1
    InvalidAlphaValue : si alpha n'est pas entre 0 et 1
    """
    check_proportion(p0)
    n = len(data)
    data_sorted = np.unique(data)
    p_emp = []

    for value in data_sorted:
        p_emp.append(np.sum(data == value) / n)

    p_emp = np.array(p_emp)
    k = len(p_emp)
    xi = n * np.sum(((p_emp - p0) ** 2) / p0)
    p_value = compute_pvalue(statspdfs.chi2, xi, np.inf, args_pdf=k - 1)
    print(f"Xi = {xi}, p value = {p_value}")
    return p_value


def test_chi2_continu(x, y, absolute_sigma, fonction, params=(), alpha=0.05):
    """
    Test du chi2 dans un cas continu avce un risque alpha.
    Goodness of fit


    Parameters :
    ------------
    x : nd array
    y : nd array
    absolute_sigma : nd array
    fonction : function
    params : tuple, paramètres de la fonction à fiter
    alpha : float, risque du test, par défaut = 0.05

    Raises:
    InvalidAlphaValue : si alpha n'est pas entre 0 et 1
    """
    check_alpha(alpha)
    # generate error if x and y don't have the same shape
    p = len(params)
    n = len(x)
    xi = np.sum(((y - fonction(x, params)) / absolute_sigma) ** 2)
    p_value = compute_pvalue(statspdfs.chi2, xi, np.inf, args_pdf=n - p)
    print(f"Chi2 = {xi}, p value = {p_value}")
    print(f"Chi2 / (N - p) = {xi / (n - p)}")
    conclude_pvalue(p_value, alpha)
    return p_value


#--------------Liste des fonctions-----------------


def List_Functions():
    print('test_proportion(data, p0, alpha=0.05, h1="two-sided")')
    print("------------ \n")
    print('test_proportion_2samples(data1, data2, alpha=0.05, h1="two-sided")')
    print("------------ \n")
    print('test_moyenne(data, mu0, alpha=0.05, h1="two-sided")')
    print("------------ \n")
    print('test_variance(data, var0, alpha=0.05, h1="two-sided")')
    print("------------ \n")
    print('test_egalite_gauss_iid(data1, data2, alpha_var=0.05, alpha_moy=0.05)')
    print("------------ \n")
    print('test_welch_gauss_iid(data1, data2, alpha=0.05, h1="two-sided")')
    print("------------ \n")
    print('test_ks_th(data, cdf_th, alpha=0.05, h1="two-sided", arguments_cdf=())')
    print("------------ \n")
    print('test_ks_2samples(data1, data2, alpha=0.05, h1="two-sided")')
    print("------------ \n")
    print('test_chi2_discret(data, p0, alpha=0.05)')
    print("------------ \n")
    print('test_chi2_continu(x, y, absolute_sigma, fonction, params=(), alpha=0.05)')
    print("------------ \n")