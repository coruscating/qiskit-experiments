"""
ESP discriminator analysis class.
"""

import numpy as np
import warnings
from qiskit.exceptions import QiskitError
from qiskit_experiments.base_analysis import BaseAnalysis, AnalysisResult
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ESPDiscriminatorAnalysis(BaseAnalysis):
    def _run_analysis(
        self, experiment_data, discriminator_type="LDA", discriminator_obj = None, 
        plot: bool = True, scaled: bool = False, **options
    ):
        """Run analysis on discriminator data.
        Args:
            experiment_data (ExperimentData): The experiment data to analyze.
            discriminator_type (str): Type of discriminator to be used in analysis. Default is LDA.
            discriminator_obj (SkLearn Classifier): Discriminator object to be used in analysis. Default is None.
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
            
            self._check_data_classes(_xdata)

            if discriminator_type == "LDA":
                discriminator[q] = LinearDiscriminantAnalysis()
            elif discriminator_type == "QDA":
                discriminator[q] = QuadraticDiscriminantAnalysis()
            elif discriminator_type == "sklearn":
                if discriminator_obj is not None:
                    self._type_check_discriminator(discriminator_obj)
                else: 
                    from sklearn.neighbors import KNeighborsClassifier
                    discriminator_obj = KNeighborsClassifier(metric='manhattan', n_neighbors=50,
                            weights='uniform')
                    warnings.warn(f'''Expected sklearn classifier object to be passed,
                                      instead {discriminator_obj} provided. The default 
                                      is set to {discriminator_obj}.''', RuntimeWarning, stacklevel=2)
    
                discriminator[q] = discriminator_obj
            
            if scaled:
                from sklearn.preprocessing import StandardScaler
                _ydata = StandardScaler(with_std=True).fit(_ydata).transform(_ydata)

            discriminator[q].fit(_ydata, _xdata)

            if plot:
                self._plot_data(_xdata, _ydata, discriminator[q], ax[q])

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
        elif discriminator_type == "sklearn":
            pass
            analysis_result = AnalysisResult(
                {
                    "discriminator": discriminator,
                    "score": score,
                    "plt": ax,
                }
            )
            
        return analysis_result, None
    
    @staticmethod
    def _type_check_discriminator(classifier):
        """ Checks whether the discriminator provided is a valid sklearn classifier with fit and 
            predict methods. """
        for name in ['fit', 'predict']:
            if not callable(getattr(classifier, name, None)):
                raise QiskitError(
                    'Discriminator of type "{}" does not have a callable "{}"'
                    ' method.'.format(type(classifier).__name__, name)
                )

    @staticmethod
    def _check_data_classes(x_data):
        """ Checks whether the data provided is a valid ESP data or not with three or more classes."""
        classes = len(np.unique(x_data))
        if classes < 3:
                raise QiskitError(
                    f'''Data corresponds to only {classes} classes. Expected 3 or more!'''
               )

    def _process_data(self, experiment_data, qubit):
        """Returns x and y data for discriminator on specific qubit."""
        xdata = np.array(
            [int(experiment_data.data[0]["metadata"]["ylabel"][qubit])]
            * len(experiment_data.data[0]["memory"])
        )
        ydata = experiment_data.data[0]["memory"][:, qubit, :]
        
        for idx in range(1, len(experiment_data.data)):
            xdata = np.concatenate(
                (
                    xdata,
                    [int(experiment_data.data[idx]["metadata"]["ylabel"][qubit])]
                    * len(experiment_data.data[idx]["memory"]),
                )
            )
            ydata = np.concatenate((ydata, experiment_data.data[idx]["memory"][:, qubit, :]))
            
        return xdata, ydata
    
    def _plot_data(self, x_data, y_data, discriminator,  ax):
        """Plots x and y data for discriminator for a specific qubit."""
        xx, yy = np.meshgrid(
            np.arange(
                min(y_data[:, 0]),
                max(y_data[:, 0]),
                (max(y_data[:, 0]) - min(y_data[:, 0])) / 500,
            ),
            np.arange(
                min(y_data[:, 1]),
                max(y_data[:, 1]),
                (max(y_data[:, 1]) - min(y_data[:, 1])) / 500,
            ),
        )
        scatter = ax.scatter(y_data[:, 0], y_data[:, 1], c=x_data)
        zz = discriminator.predict(np.c_[xx.ravel(), yy.ravel()])
        zz = np.array(zz).astype(float).reshape(xx.shape)
        ax.contourf(xx, yy, zz, alpha=0.2)
        ax.set_xlabel("I data")
        ax.set_ylabel("Q data")
        ax.legend(*scatter.legend_elements())
        
