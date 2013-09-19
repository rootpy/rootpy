import string
from rootpy.plotting.contrib import plot_contour_matrix


if __name__ == '__main__':

    import numpy as np

    n_vars = 5
    var_names = ['var_%s' % s for s in string.lowercase[:n_vars]]

    def random_symm(n):
        a = np.random.random_integers(-10, 10, size=(n, n))
        return (a + a.T) / 2

    data_a = np.random.multivariate_normal(
        -np.random.random(n_vars) * 3, cov=random_symm(n_vars), size=100000)
    data_b = np.random.multivariate_normal(
        np.random.random(n_vars) * 3, cov=random_symm(n_vars), size=100000)

    plot_contour_matrix(
        [data_a, data_b],
        var_names,
        'out.gif',
        sample_names='A B'.split(),
        sample_colors='red blue'.split(),
        sample_lines='solid dashed'.split(),
        num_contours=5,
        cell_width=2,
        cell_height=2,
        animate_field='var_a',
        animate_steps=20)
