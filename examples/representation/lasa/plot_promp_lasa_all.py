from plot_promp_lasa import load,learn
import matplotlib.pyplot as plt

numWeights = 30 #how many weights shall be used 

def draw(x,y,traj,idx,axs):
    h = int((2*idx)/6)
    w = int((2*idx)%6)
    axs[h,w].plot(y.transpose(2,1,0)[0], y.transpose(2,1,0)[1])
    
    mean, _ , covar = traj.trajectory()
    axs[h,w+1].plot(mean[:,0],mean[:,1])
    traj.plotCovariance(axs[h,w+1],mean,covar.reshape(-1,4,4))
        
    axs[h,w+1].set_xlim(axs[h,w].get_xlim())
    axs[h,w+1].set_ylim(axs[h,w].get_ylim())
    axs[h,w].get_yaxis().set_visible(False)
    axs[h,w].get_xaxis().set_visible(False)
    axs[h,w+1].get_yaxis().set_visible(False)
    axs[h,w+1].get_xaxis().set_visible(False)

def main():
	numShapes = 30
	width = 6
	height = numShapes*2/width
	fig, axs = plt.subplots(int(height),int(width))

	for i in range(numShapes):
	    x,y = load(i)
	    traj = learn(x,y)
	    draw(x,y,traj,i,axs)
	plt.show()

if __name__ == "__main__":
    main()

