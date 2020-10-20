import numpy as np

class HiddenMarkov:
    def __init__(self):
        self.alphas = None
        self.forword_p = None
        self.beats = None
        self.backward_p = None

