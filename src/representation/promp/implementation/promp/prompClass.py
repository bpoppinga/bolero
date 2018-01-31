import promp
import numpy as np
from matplotlib.patches import Ellipse 


class Promp:
    data = None
    def __init__(self, numBF,numDim,isStroke,overlap = 0.5):
        self.data = promp.TrajectoryData(numBF,numDim,isStroke,overlap)
    
    def imitate(self,x,y):
        assert(len(x) == len(y))
	sizes_ = []	
	x_ = []
	y_ = []	
	for i in range(len(x)):
        	assert(len(x[i]) == len(y[i]))
		sizes_ += [x[i].shape[0]]
		x_ += x[i].flatten().tolist()
        	y_ += y[i].flatten().tolist()
        self.data.imitate(np.array(sizes_,float),np.array(x_),np.array(y_)) 
        
    def step(self, timestamp):
        valueMeans = np.empty(2*self.data.num_dim_)
        valueCovs = np.empty((2*self.data.num_dim_)**2)
        self.data.step(timestamp,valueMeans,valueCovs)
        valueCovs = valueCovs.reshape(2*self.data.num_dim_,2*self.data.num_dim_)
        return (valueMeans,valueCovs)
    
    def sample(self):
        ret = Promp(self.data.num_b_f_,self.data.num_dim_,self.data.is_stroke_)
        self.data.sample_trajectory_data(ret.data)
        return ret 
    
    def getValues(self,timestamps):
        means = np.empty((self.data.num_dim_*2*len(timestamps)))
        covars = np.empty((self.data.num_dim_*2*self.data.num_dim_*2*len(timestamps)))
        self.data.get_values(timestamps,means,covars)
        means = means.reshape((self.data.num_dim_*2,-1)).transpose(1,0)
        covars = covars.reshape((-1,self.data.num_dim_*2,self.data.num_dim_*2))
        return (means,covars)
    
    def condition(self,conditionPoints):
        assert len(conditionPoints) >= 1
        assert len(conditionPoints[0]) == 5
        conditionPoints_ = conditionPoints.flatten()
        self.data.condition(len(conditionPoints),conditionPoints_)

    def plot(self,plot, numSamples = 0, useCovar = True, samples = 1000, start = 0, end = 1, nstd = 2):

        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]

        means, covariances = self.getValues(np.linspace(start,end,samples))
        
        for i in range(numSamples):
            # instead off covariance ellipse draw multiple samples
            values , _ = self.sample().getValues(np.linspace(0,1,100))
            plot.plot( values[:,0],values[:,2])   

        if useCovar:
            for k in range(len(means)):
                mean = [ means[:,0][k],   means[:,2][k]]
                
                cov = np.empty((2,2))
                cov[0,0] = covariances[k,0,0]
                cov[0,1] = covariances[k,0,2]
                cov[1,0] = covariances[k,2,0]
                cov[1,1] = covariances[k,2,2]
              
                vals, vecs = eigsorted(cov)
                theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                
                width, height = 2 * nstd * np.sqrt(vals)
                
                ell = Ellipse(xy=mean, width=width, height=height, angle = theta,alpha=1, color='grey')
                plot.add_patch(ell) 
            plot.plot(means[:,0],means[:,2],color='red')


class CombinedPromp:
    data = None
    def __init__(self):
        self.data = promp.CombinedTrajectoryData()
    
    def add(self,trajectory, activation):
        self.data.add_trajectory(trajectory.data,activation.flatten())
    
    def step(self, timestamp):
        valueMeans = np.empty(2*self.data.num_dim_)
        valueCovs = np.empty((2*self.data.num_dim_)**2)
        self.data.step(timestamp,valueMeans,valueCovs)
        valueCovs = valueCovs.reshape(2*self.data.num_dim_,2*self.data.num_dim_)
        return (valueMeans,valueCovs)
    
    def getValues(self,timestamps):
        means = np.empty((self.data.num_dim_*2*len(timestamps)))
        covars = np.empty((self.data.num_dim_*2*self.data.num_dim_*2*len(timestamps)))
        self.data.get_values(timestamps,means,covars)
        means = means.reshape((self.data.num_dim_*2,-1)).transpose(1,0)
        covars = covars.reshape((-1,self.data.num_dim_*2,self.data.num_dim_*2))
        return (means,covars)

    def plot(self,plot, useCovar = True, samples = 1000, start = 0, end = 1, nstd = 2):

        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]

        means, covariances = self.getValues(np.linspace(start,end,samples))
        
        
        if useCovar:
            for k in range(len(means)):
                mean = [ means[:,0][k],   means[:,2][k]]
                
                cov = np.empty((2,2))
                cov[0,0] = covariances[k,0,0]
                cov[0,1] = covariances[k,0,2]
                cov[1,0] = covariances[k,2,0]
                cov[1,1] = covariances[k,2,2]
              
                vals, vecs = eigsorted(cov)
                theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                
                width, height = 2 * nstd * np.sqrt(vals)
                
                ell = Ellipse(xy=mean, width=width, height=height, angle = theta,alpha=1, color='grey')
                plot.add_patch(ell) 
            plot.plot(means[:,0],means[:,2],color='red')
        
    
