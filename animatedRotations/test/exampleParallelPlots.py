import multiprocessing
import numpy as np

def plot(args):
	#import multiprocessing in each worker
	import matplotlib.pyplot as plt
	data = np.random.rand(500,500)
	DPI=300
	vmin=-100
	vmax=100
	num, i = args
	fig, ax = plt.subplots()
	im = ax.imshow(data,cmap="jet",vmin=vmin,vmax=vmax)
	ax.invert_yaxis()
	plt.colorbar(im,ax=ax)
	plt.title('Plot of a %i' % num)
	fig.savefig('temp_fig_%02i.png' % i)
	plt.close()
	return None


### try doing this asynchronously to save on memory
pool = multiprocessing.Pool(5)
num_figs = 20
args= zip(np.random.randint(10,1000,num_figs), 
			range(num_figs))
print(pool.map(plot, args))
#pool.close()
#pool.join()
print("pool closed and process competed!")
	


