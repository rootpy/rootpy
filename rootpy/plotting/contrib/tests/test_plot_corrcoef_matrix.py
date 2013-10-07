import string
from rootpy.plotting.contrib import plot_corrcoef_matrix


if __name__ == '__main__':

    import numpy as np

    n_vars = 10
    var_names = ['var_%s' % s for s in string.lowercase[:n_vars]]

    def random_symm(n):
        a = np.random.random_integers(-10, 10, size=(n, n))
        return (a + a.T) / 2

    data = np.random.multivariate_normal(
        -np.random.random(n_vars) * 3, cov=random_symm(n_vars), size=100000)
    weights = np.random.randint(1, 10, 100000)

    plot_corrcoef_matrix(
        data, var_names, 'correlations.png',
        weights=weights, title='correlations')
