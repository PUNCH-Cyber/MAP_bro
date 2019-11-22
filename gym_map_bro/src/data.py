import numpy as np
import pandas as pd

class metaData(object):
    def __init__(self,val = pd.Series([0]),val_tot = 0,ind = 0,rplan = [1,2,3,0]):
        self.val = val
        self.val_tot = val_tot
        self.ind = ind
        self.rplan = rplan

#    def save(self,val,val_arg):
#        self.val[val_arg] = val
#        return

class dataItem(object):
    def __init__(self,data = pd.Series([0]) ,val = pd.Series([0]),val_tot =0, ind = 0,rplan = [1,2,3,0]):
        self.data = data
        self.val = val
        self.val_tot = val_tot
        self.ind = int(ind)
        self.rplan = rplan
        self.metaData = metaData(val,val_tot,int(ind),rplan)


class dataBatch(object):
    def __init__(self,data,val,val_tot,ind,rplan):
        self.batch = []
        self.size = len(ind)
        for i in np.arange(self.size):
            self.batch.append(dataItem(data.iloc[i],val.iloc[i],val_tot[i],ind[i],rplan[i]))

    def get(self, variable, md=0): # Method for gathering all of a certain variable in the entire batch
        if isinstance(self.batch[0].__dict__[variable], pd.Series):
            var = pd.DataFrame(columns = self.batch[0].__dict__[variable].columns)
            for i in np.arange(self.size):
                if md == 0:
                    var = var.append(self.batch[i].__dict__[variable], ignore_index=True)
                else:
                    var = var.append(self.batch[i].metaData.__dict__[variable], ignore_index=True)
        else:
            var = []
            for i in np.arange(self.size):
                if md == 0:
                    var.append(self.batch[i].__dict__[variable])
                else:
                    var.append(self.batch[i].metaData.__dict__[variable])
        return var

    def age_step(self):
        for i in np.arange(self.size):
            self.batch[i].val[0] += 1
