import matplotlib.pyplot as plt
import ipywidgets as widgets
import numpy as np

from io import BytesIO
from ipywidgets import interact

class CurveGenerator():
    def __init__(self, a=3, b=2.2):
        self.a = a
        self.b = b
        self.cb = None
    
    def refresh(self, a, b):
        self.a = a
        self.b = b

        ax = plt.axes()
        
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])

        x = np.linspace(0, 1, 100)
        ax.plot(x, self.evaluate(x))
    
    def evaluate(self, x):
        return np.power(x, self.a) / ( np.power(x, self.a) + np.power(self.b - self.b * x, self.a) )

    def gui(self):
        return interact(self.refresh,
            a=widgets.FloatSlider(value=self.a, min=0, max=10),
            b=widgets.FloatSlider(value=self.b, min=0, max=10))
