
class FScore(object):
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
    def add_tp(self, tp=1):
        self.tp += tp
    
    def add_fp(self, fp=1):
        self.fp += fp

    def add_tn(self, tn=1):
        self.tn += tn

    def add_fn(self, fn=1):
        self.fn += fn

    def precision(self):
        if self.tp == 0:
            return 0.0
        return float(self.tp) / (self.tp + self.fp)

    def recall(self):
        if self.tp == 0:
            return 0.0
        return float(self.tp) / (self.tp + self.fn)

    def f1(self):
        precision = self.precision()
        recall = self.recall()
        if (precision + recall) == 0.0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

