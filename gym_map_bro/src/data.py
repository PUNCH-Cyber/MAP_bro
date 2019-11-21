import pandas as pd

class metaData(object):
    def __init__(self,val = pd.Series([0]),ind = 0,rplan = [1,2,3,0]):
        self.val = val
        self.ind = ind
        self.rplan = rplan

    def save(self,val,val_arg):
        self.val[val_arg] = val
        return

class dataItem(object):
    def __init__(self,data = pd.Series([0]) ,val = pd.Series([0]), ind = 0,rplan = [1,2,3,0]):
        self.data = data
        self.val = val
        self.ind = ind
        self.rplan = rplan
        self.metaData = metaData(val,ind,rplan)

class dataBatch(object):
    def __init__(self,data,val,ind,rplan):
        self.batch = []
        for i in np.arange(len(ind)):
            self.batch.append(dataItem(data,val,ind,rplan))