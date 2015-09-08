"""
Plot function reads in data from Data.csv where first column is x-axis and the remaining columns are y-axis data, file should contain a header row
"""

ColorDict = {}
ColorDict['Amikacin']='RoyalBlue'
ColorDict['Ampicillin']='FireBrick'
ColorDict['Ciprofloxacin']='OliveDrab'
ColorDict['Colistin']='DarkCyan'
ColorDict['Doxycycline']='Coral'
ColorDict['Erythromycin']='CornflowerBlue'
ColorDict['IPTG']='Tomato'
ColorDict['Rifampicin']='YellowGreen'
ColorDict['Tetracycline']='DarkViolet'
ColorDict['Vancomycin']='DarkTurquoise'
ColorDict['Potassium Nitrate']='SaddleBrown'

import numpy as np
import matplotlib.pyplot as plt

plt.ioff()

#--------------------------------------------------------------------------------------
# Plotting several Raman spectra pairs in individual boxes (i.e., for comparing spontaneous and SERS spectra)

def specpairs():

	Data = np.loadtxt('Data.csv', delimiter=',', skiprows=1)
	Labels = ['Tetracycline','Doxycycline','Rifampicin','Colistin']

	xx = Data[:,0].reshape((Data.shape[0],1))
	yy = Data[:,1:]
	nPlots = 4  # Data.shape[1]-1

	plt.rc('font', family = 'Arial', size='12')

	# Subplots with some shared axes
	f, Axlst = plt.subplots(nPlots, sharex=True, sharey=False, figsize=(6,9))

	for i in range(nPlots):
		Axlst[i].plot(xx,yy[:,2*i])
		Axlst[i].plot(xx,yy[:,2*i+1])
		Axlst[i].set_yticks(np.arange(0.2, 1.8, 0.2))
		Axlst[i].set_yticklabels([1],visible=False)
		if i == 3:
			Axlst[i].set_ylim(0,1.7)
		else:
			Axlst[i].set_ylim(0,1.1)
		Axlst[i].set_title(Labels[i],ha='left',x=0.05,y=0.8)

	
	# Axlst[0].set_title('Sharing both axes')
	Axlst[-1].set_xlabel(r'Wavenumber ($\mathregular{cm}^{-1}$)')
# 	Axlst[-1].set_ylabel('Normalized Intensity (a.u.)')
	
	f.text(0.09,0.5,'Normalized Intensity (a.u.)',rotation=90,ha='center',va='center')
	
	# Fine-tune figure; make subplots close to each other and hide x ticks for all but bottom plot.
	f.subplots_adjust(hspace=0)
	plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

	plt.show()
	
	
#--------------------------------------------------------------------------------------
# Plotting several Raman spectra in a single box plot
	
def specsinglebox():
	import numpy as np
	import matplotlib.pyplot as plt

	f = open('Data1.csv', mode='r')
	header = f.readline()
	
	Labels = header.replace('\n','').replace('Wavenumber,','').split(',')
	Data = np.loadtxt('Data1.csv', delimiter=',', skiprows=1)

	xx = Data[:,0].reshape((Data.shape[0],1))
	yy = Data[:,1:]
	nPlots = yy.shape[1]

	plt.ioff()

	plt.rc('font', family = 'Arial', size='12')
	
# Subplots with some shared axes
	f, Axlst = plt.subplots(1, sharex=True, sharey=False, figsize=(10,6))

	for i in range(nPlots):
		Axlst.plot(xx,yy[:,nPlots-1-i],\
		color=ColorDict[Labels[nPlots-1-i]],label=Labels[nPlots-1-i])
		
	Axlst.set_xlabel(r'Wavenumber ($\mathregular{cm}^{-1}$)')
	Axlst.set_ylabel('Normalized Intensity (a.u.)')
	Axlst.set_yticks(np.arange(0, np.max(yy), 0.5))
	Axlst.set_yticklabels([1],visible=False)
	
# Shrink current axis by 20%
	box = Axlst.get_position()
	Axlst.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	Axlst.legend(bbox_to_anchor=(1,0.5),loc='center left')

	plt.show()
	
# specsinglebox()
