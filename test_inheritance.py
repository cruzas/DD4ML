from abc import abstractmethod

class ParallelizedModel:
    def __init__(self):
        self.a = 1
        self.b = 2
    
    def print(self):
        print(self.a, self.b)
    
    def random_fun(self):
        print('random_fun')

class WeightParallelizedModel(ParallelizedModel):
    def __init__(self):
        super().__init__()
        self.c = 3
        self.d = 4

    def print(self):
        print(self.a, self.b, self.c, self.d)

p = ParallelizedModel()
c = WeightParallelizedModel()

c.a = 100
c.b = 123
c.random_fun()
c.blah()

p.print()
c.print()


