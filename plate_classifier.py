import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from sklearn.metrics import mean_squared_error

from utils import *

class PlateClassifier():
    def __init__(self, input_size=(64,64), connectivity=4):
        self.input_size = input_size
        self.connectivity = connectivity

    def fit(self, X_train, y_train, stats=None):
        self.data = X_train
        self.alf = y_train
        self.stats = stats

    def split(self, img_plate, verbose=False):
        img_plate = cv2.cvtColor(img_plate, cv2.COLOR_BGR2RGB)

        # deixando a imagem em gray e binária
        gray = cv2.cvtColor(img_plate, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        _, thresh = cv2.threshold(
            gray,
            127, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # separando os componentes
        connectivity = self.connectivity
        num_labels, cc_image, stats, centroids = cv2.connectedComponentsWithStats(
            thresh , connectivity , cv2.CV_32S
        )

        # iterando sob os componentes
        components = []
        for i in range(num_labels):
            x, y, w, h = stats[i][:4]
            
            component = gray[y:y+h, x:x+w]

            # tornando o componente binário
            _, c_thresh = cv2.threshold(component, 127, 255, cv2.THRESH_BINARY)
            component = cv2.resize(c_thresh,self.input_size)
            
            components.append(c_thresh)

        if verbose:
            print('num components =', len(components))
            print('components')
            plot_sidebyside(
                components,
                [str(i) for i in range(len(components))],
                colormap='gray'
            )

        return components

    def predict(self, X_test, split=True, verbose=False, show_scores=False):
        """
        X_test: imagem da placa ou lista de componentes
        split: caso 'True', separa a placa em componentes
        """
        if split:
            components = self.split(X_test, verbose=verbose)
        else:
            components = X_test

        # texto final de output
        text = ''

        # realizando classificação caractere a caractere
        for c, i in zip(components, range(len(components))):

            if verbose:
                print('\ncomponent', i+1, 'of', len(components))

            # classificação
            pred = self.predict_char(c, verbose=verbose, show_scores=show_scores)

            # exibindo a previsão do caractere
            if verbose:
                print('pred =', pred)

            text += pred
        return text

    def predict_char(self, X_test, verbose=False, show_scores=False, return_idx=False):
        X_test = cv2.resize(X_test, (64,64));
        _, X_test = cv2.threshold(X_test, 127, 255, cv2.THRESH_BINARY_INV)

        metrics = []
        # obtendo score para cada letra template da base
        for template in self.data:
            _, template = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)

            # intersection
            inter = X_test + template
            _, inter = cv2.threshold(inter, 255, 255, cv2.THRESH_BINARY)
            inter = len(inter.ravel()[inter.ravel() == 255])

            # extra input
            extrai = X_test - template
            _, extrai = cv2.threshold(extrai, 127, 255, cv2.THRESH_BINARY)
            extrai = len(extrai.ravel()[extrai.ravel() == 255])

            # extra template
            extrat = template - X_test
            _, extrat = cv2.threshold(extrat, 127, 255, cv2.THRESH_BINARY)
            extrat = len(extrat.ravel()[extrat.ravel() == 255])

            score = (extrai + extrat) - inter
            metrics.append(score)
        
        idx = np.argmax(metrics)
        metrics = pd.Series(data=metrics, index=self.alf)

        # exibindo images obtidas no processo
        if verbose:

            template = self.data[idx]

            best_fit = cv2.threshold(self.data[idx], 127, 255, cv2.THRESH_BINARY_INV)[1]
            diff = cv2.threshold(self.data[idx], 127, 255, cv2.THRESH_BINARY_INV)[1] - X_test

            inter = X_test + template
            _, inter = cv2.threshold(inter, 255, 255, cv2.THRESH_BINARY)

            extrai = X_test - template
            _, extrai = cv2.threshold(extrai, 127, 255, cv2.THRESH_BINARY)

            extrat = template - X_test
            _, extrat = cv2.threshold(extrat, 127, 255, cv2.THRESH_BINARY)

            plot_sidebyside(
                [X_test, best_fit, diff, inter, extrai, extrat],
                ['input', 'best fit', 'diff', 'intersection', 'input extra', 'template extra'],
                colormap='gray'
            )

            # exibindo ranking
            if show_scores:
                print(metrics.sort_values(ascending=False), end='\n')

        if return_idx:
            return idx
        else:
            return self.alf[idx]

    def _predict(self,
                 X_test,
                 function=mean_squared_error,
                 metrics_callable=get_image_stats,
                 templates=None,
                 verbose=False):
        if verbose:
            plot_sidebyside(
                X_test,
                [str(i) for i in range(len(X_test))]
            )

        output = ''
        for component, i in zip(X_test, range(len(X_test))):

            if verbose:
                print('component', i)
                plt.imshow(component);
                plt.show();

            char = self._predict_char(
                char_image=component,
                function=function,
                metrics_callable=metrics_callable,
                templates=templates,
                verbose=verbose
            )

            if verbose:
                print('pred =', char, end='\n\n')

            output += char
        return output

    def _predict_char(self,
                      char_image,
                      function=mean_squared_error,
                      metrics_callable=get_image_stats,
                      templates=None,
                      verbose=False):
        stats = metrics_callable(char_image, templates=templates)

        mse = self.stats.apply(
            lambda x: function(x.values, stats),
        axis=1
        )

        mse = pd.Series(data=mse, index=self.stats.index)

        if verbose:
            print(mse.sort_values(ascending=True))

        return mse.sort_values(ascending=True).index[0]