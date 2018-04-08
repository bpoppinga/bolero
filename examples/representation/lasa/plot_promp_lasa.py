import numpy as np
from lasa import load_lasa
import sys
from bolero.representation import ProMPBehavior
import matplotlib.pyplot as plt

numWeights = 30 #how many weights shall be used 

def load(idx):    
    X, Xd, Xdd, dt, shape_name = load_lasa(idx)
    y = X.transpose(2,1,0)	    
    x = np.linspace(0,1,1000)
    return (x,y)

def learn(x,y):
    traj = ProMPBehavior(1.0, 1.0/999.0, numWeights,learnCovariance=True,useCovar=True)
    traj.init(4,4)
    traj.imitate(y.transpose(2,1,0))  
    return traj

def draw(x,y,traj,useCovariance,useSamples):
	ax = plt.subplot(121)
	ax.plot(y.transpose(2,1,0)[0], y.transpose(2,1,0)[1])
	bx = plt.subplot(122)
	bx.set_xlim(ax.get_xlim())
	bx.set_ylim(ax.get_ylim())	

	mean, _ , covar = traj.trajectory()
        bx.plot(mean[:,0],mean[:,1])
        if useCovariance:
            ProMPBehavior.plotCovariance(bx,mean,covar.reshape(-1,4,4))	
	

def main(argv):
	""" prints one lasa example with covariance or samples
		arg 0: number of the example
        arg 1: use covariance (0 = false)
        arg 2: use samples (0 = false)
    """
	useCovariance = bool(int(argv[1]))
	useSamples = bool(int(argv[2]))
	x,y = load(int(argv[0]))
	traj = learn(x,y)
	draw(x,y,traj,useCovariance,useSamples)
	plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])


