"""
Standard discriminator analysis class.
"""

import numpy as np
from qiskit_experiments.base_analysis import BaseAnalysis, AnalysisResult
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class DiscriminatorAnalysis(BaseAnalysis):
    def _run_analysis(
        self, experiment_data, discriminator_type="LDA", plot: bool = True, **options
    ):
        """Run analysis on discriminator data.
        Args:
            experiment_data (ExperimentData): The experiment data to analyze.
            discriminator_type (str): Type of discriminator to use in analysis. Default is LDA.
            options: kwarg options for analysis function.
        Returns:
            tuple: A pair ``(analysis_results, figures)`` where
                ``analysis_results`` may be a single or list of
                AnalysisResult objects, and ``figures`` may be
                None, a single figure, or a list of figures.
        """

        nqubits = len(experiment_data.data[0]["metadata"]["ylabel"])
        discriminator = [None] * nqubits
        score = [None] * nqubits
        fig, ax = plt.subplots(nqubits)
        fig.tight_layout()
        if nqubits == 1:
            ax = [ax]

        for q in range(nqubits):
            _xdata, _ydata = self._process_data(experiment_data, q)

            if discriminator_type == "LDA":
                discriminator[q] = LinearDiscriminantAnalysis()
            elif discriminator_type == "QDA":
                discriminator[q] = QuadraticDiscriminantAnalysis()

            discriminator[q].fit(_ydata, _xdata)

            if plot:
                xx, yy = np.meshgrid(
                    np.arange(
                        min(_ydata[:, 0]),
                        max(_ydata[:, 0]),
                        (max(_ydata[:, 0]) - min(_ydata[:, 0])) / 500,
                    ),
                    np.arange(
                        min(_ydata[:, 1]),
                        max(_ydata[:, 1]),
                        (max(_ydata[:, 1]) - min(_ydata[:, 1])) / 500,
                    ),
                )
                scatter = ax[q].scatter(_ydata[:, 0], _ydata[:, 1], c=_xdata)
                zz = discriminator[q].predict(np.c_[xx.ravel(), yy.ravel()])
                zz = np.array(zz).astype(float).reshape(xx.shape)
                ax[q].contourf(xx, yy, zz, alpha=0.2)
                ax[q].set_xlabel("I data")
                ax[q].set_ylabel("Q data")
                ax[q].legend(*scatter.legend_elements())
            score[q] = discriminator[q].score(_ydata, _xdata)

        if discriminator_type == "LDA":
            analysis_result = AnalysisResult(
                {
                    "discriminator": discriminator,
                    "coef": [d.coef_ for d in discriminator],
                    "intercept": [d.intercept_ for d in discriminator],
                    "score": score,
                    "plt": ax,
                }
            )

        elif discriminator_type == "QDA":
            analysis_result = AnalysisResult(
                {
                    "discriminator": discriminator,
                    "rotations": [d.rotations_ for d in discriminator],
                    "score": score,
                    "plt": ax,
                }
            )

        return analysis_result, None

    def _process_data(self, experiment_data, qubit):
        """Returns x and y data for discriminator on specific qubit."""
        xdata = np.array(
            [int(experiment_data.data[0]["metadata"]["ylabel"][qubit])]
            * len(experiment_data.data[0]["memory"])
        )
        ydata = experiment_data.data[0]["memory"][:, qubit, :]
        xdata = np.concatenate(
            (
                xdata,
                [int(experiment_data.data[1]["metadata"]["ylabel"][qubit])]
                * len(experiment_data.data[1]["memory"]),
            )
        )
        ydata = np.concatenate((ydata, experiment_data.data[1]["memory"][:, qubit, :]))
        return xdata, ydata
    
    def dataset_fixed_cov():
        '''Generate 2 Gaussians samples with the same covariance matrix'''
        n, dim = 300, 2
        np.random.seed(0)
        C = np.array([[0., -0.23], [0.83, .23]])
        X = np.r_[np.dot(np.random.randn(n, dim), C),
                  np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
        y = np.hstack((np.zeros(n), np.ones(n)))
        return X, y

    def dataset_cov():
        '''Generate 2 Gaussians samples with different covariance matrices'''
        n, dim = 300, 2
        np.random.seed(0)
        C = np.array([[0., -1.], [2.5, .7]]) * 2.
        X = np.r_[np.dot(np.random.randn(n, dim), C),
                  np.dot(np.random.randn(n, dim), C.T) + np.array([1, 4])]
        y = np.hstack((np.zeros(n), np.ones(n)))
        return X, y

    def plot_data(lda, X, y, y_pred, fig_index):
        splot = plt.subplot(2, 2, fig_index)
        if fig_index == 1:
            plt.title('no_QDA')
            plt.ylabel('Data with\n fixed covariance')
        elif fig_index == 2:
            plt.title('QDA')
        elif fig_index == 3:
            plt.ylabel('Data with\n varying covariances')

        tp = (y == y_pred)  # True Positive
        tp0, tp1 = tp[y == 0], tp[y == 1]
        X0, X1 = X[y == 0], X[y == 1]
        X0_tp, X0_fp = X0[tp0], X0[~tp0]
        X1_tp, X1_fp = X1[tp1], X1[~tp1]

        # class 0: dots
        plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker='.', color='red')
        plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker='x',
                    s=20, color='#990000')  # dark red

        # class 1: dots
        plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='blue')
        plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='x',
                    s=20, color='#000099')  # dark blue

        # class 0 and 1 : areas
        nx, ny = 200, 100
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                             np.linspace(y_min, y_max, ny))
        Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 1].reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                       norm=colors.Normalize(0., 1.), zorder=0)
        plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')

        # means
        plt.plot(lda.means_[0][0], lda.means_[0][1],
                 '*', color='yellow', markersize=15, markeredgecolor='grey')
        plt.plot(lda.means_[1][0], lda.means_[1][1],
                 '*', color='yellow', markersize=15, markeredgecolor='grey')

        return splot

    def LinearDiscriminator(X,y,dataset_fixed_cov(),dataset_cov()):
        for i, (X, y) in enumerate([dataset_fixed_cov(), dataset_cov()]):
            X0 = X[y==0]
            X1 = X[y==1]
            X0_mean=np.mean(X0,axis=0).reshape(1,2)
            X1_mean=np.mean(X1,axis=0).reshape(1,2)
            N_zeros=len(X0)
            N_ones=len(X1)
            K=2
            sigma_X0 = np.dot((X0-X0_mean).T,(X0-X0_mean))/(N_zeros-K)
            sigma_X1 = np.dot((X1-X1_mean).T,(X1-X1_mean))/(N_ones-K)
            sigma = sigma_X0 + sigma_X1
            sigma_inv = inv(sigma)
            RHS = 0.5 * np.matmul(np.matmul((X1_mean+X0_mean),sigma_inv),(X1_mean-X0_mean).T) - log(N_ones/N_zeros)
            LHS = np.matmul(np.matmul(X,sigma_inv),(X1_mean-X0_mean).T)
            a = []
            for i in range(len(LHS)):
                if (LHS[i]>RHS):
                    a.append(1)
                else :
                    a.append(0)
        return (a)
    
    
    def QuadraticDiscriminator(X,y,dataset_fixed_cov(),dataset_cov()):
        for i, (X, y) in enumerate([dataset_fixed_cov(), dataset_cov()]):
            X0 = X[y==0]
            X1 = X[y==1]
            X0_mean=np.mean(X0,axis=0).reshape(1,2)
            X1_mean=np.mean(X1,axis=0).reshape(1,2)
            N_zeros=len(X0)
            N_ones=len(X1)
            K=2
            sigma_X0 = np.dot((X0-X0_mean).T,(X0-X0_mean))/(N_zeros-K)
            sigma_X1 = np.dot((X1-X1_mean).T,(X1-X1_mean))/(N_ones-K)
            sigma_inv = inv(sigma)
            log_sigma_X0 = np.log(np.linalg.det(sigma_X0))
            log_sigma_X1 = np.log(np.linalg.det(sigma_X1))
            a=[]
            for p in range(X.shape[0]):
                delta_0 = -0.5*log_sigma_X0 - 0.5*np.matmul(np.matmul((X[p]-X0_mean),inv(sigma_X0)),(X[p]-X0_mean).T) + 0.5 
                delta_1 = -0.5*log_sigma_X1 - 0.5*np.matmul(np.matmul((X[p]-X1_mean),inv(sigma_X1)),(X[p]-X1_mean).T) + 0.5 
                if (delta_0>delta_1):
                    a.append(0)
                else :
                    a.append(1)
         return (a)
