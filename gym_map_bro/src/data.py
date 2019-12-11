import numpy as np
import pandas as pd

class metaData(object):
    def __init__(self,val = pd.Series([0]),val_tot = np.nan,ind = 0,rplan = [1,2,3,0]):
        self.val = val
        self.val_tot = val_tot
        self.ind = ind
        self.rplan = rplan

#    def save(self,val,val_arg):
#        self.val[val_arg] = val
#        return

class dataItem(object):
    def __init__(self,data = pd.Series([0]) ,val = pd.Series([0]),val_tot = np.nan, ind = 0,rplan = [1,2,3,0]):
        self.data = data
        self.val = val          # Vals stay constant regardless of what dataStore they are in.
        self.val_tot = val_tot  # Val_tots are saved with associated frac's and weights
        self.ind = int(ind)
        self.rplan = rplan
        self.metaData = metaData(val,val_tot,int(ind),rplan)


class dataBatch(object):
    def __init__(self,data,val,val_tot,ind,rplan):
        self.batch = []
        self.size = len(ind)
        self.columns = data.columns
        for i in np.arange(self.size):
            self.batch.append(dataItem(data.iloc[i],val.iloc[i],val_tot[i],ind[i],rplan[i]))

    def get(self, variable, md=0): # Method for gathering all of a certain variable in the entire batch
        if isinstance(self.batch[0].__dict__[variable], pd.Series):
            var = pd.DataFrame(columns = self.columns)
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

    def save(self,di,arg,vf,action):
        self.batch[arg] = di
        self.batch[arg].val_tot = vf(di.val)
        self.batch[arg].ind = [x for x in range(len(di.rplan)) if di.rplan[x] == action][0]

    def add(self,dis):
        for i in np.arange(len(dis)):
            self.batch.append(dis[i])
        self.size = len(self.batch)

    def age_step(self, vf):
        for i in np.arange(self.size):
            self.batch[i].val[0] += 1
            if not np.isnan(self.batch[i].val_tot): # don't want to update nan's to 0's
                self.batch[i].val_tot = vf(self.batch[i].val) # update val_tot with new vals
