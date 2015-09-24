"""
Read in data from 'datacsv' where first column is x-axis and the remaining columns are y-axis data. 

File should contain a header row with labels for x column first then the appropriate y columns.
"""

import argparse, re
import numpy as np
import matplotlib.pyplot as plt

# Turn off pyplot interactive mode
plt.ioff()


"""
Begin function definitions
"""


#--------------------------------------------------------------------------------------
def colordef(Labels):

	"""
	Define colors to use for each label. Must be set manually in makeplot.py.
	"""

	ColorDict = {}

	ColorDict['Amikacin']='RoyalBlue'
	ColorDict['Ampicillin']='FireBrick'
	ColorDict['Ciprofloxacin']='OliveDrab'
	ColorDict['Colistin']='DarkCyan'
	ColorDict['Doxycycline']='Coral'
	ColorDict['Erythromycin']='CornflowerBlue'
	ColorDict['IPTG']='SaddleBrown'
	ColorDict['Rifampicin']='YellowGreen'
	ColorDict['Tetracycline']='DarkViolet'
	ColorDict['Vancomycin']='DarkTurquoise'
	ColorDict['Potassium Nitrate'] = 'GoldenRod'
	ColorDict['Sodium Chloride'] = 'GoldenRod'

	ColorScheme = ['RoyalBlue', 'FireBrick', 'OliveDrab', 'DarkCyan', 'Coral', 'CornflowerBlue', 'SaddleBrown', 'YellowGreen', 'DarkViolet', 'DarkTurquoise']


	return(ColorDict, ColorScheme)
	


#--------------------------------------------------------------------------------------	
def specsinglebox(Data, Labels):
	
	"""
	Plotting several Raman spectra in a single box plot. 
	"""
	
	plt.rc('font', family = 'Arial', size='16')


	ColorDict, _ = colordef(Labels)
# 	Flip y-data and labels left-right to change plot order if needed
	Data[:,1:] = np.fliplr(Data[:,1:])
	Labels = Labels[1:]
	Labels = Labels[::-1]


	xx = Data[:,0].reshape((Data.shape[0],1))
	yy = Data[:,1:]
	nPlots = len(Labels)
	
	
# Subplots with some shared axes
	fig, Axlst = plt.subplots(1, sharex=True, sharey=False, figsize=(12,7))
	
# Set margins
	plt.subplots_adjust(bottom=0.10,left=0.05,right=0.89,top=0.95,wspace=0.1,hspace=0.1)
		
	for i in range(nPlots):
		Axlst.plot(xx,yy[:,i], label=Labels[i], color=ColorDict[Labels[i]])
		
	Axlst.set_xlabel(r'Wavenumber ($\mathregular{cm}^{-1}$)')
	Axlst.set_ylabel('Normalized Intensity (a.u.)')
# 	Axlst.set_yticks(np.arange(0, np.max(yy), 0.5))
	Axlst.set_yticklabels([1],visible=False)
	
	
# Shrink current axis by 20%
	box = Axlst.get_position()
	Axlst.set_position([box.x0, box.y0, box.width * 0.8, box.height])


# Place legend outside of axes box
	Axlst.legend(bbox_to_anchor=(1,0.5),loc='center left')
# 	Axlst.legend(loc='best')
	
# 	fig.tight_layout()
# 	fig.savefig('test.png')	

	plt.show()




#--------------------------------------------------------------------------------------
def specmultibox(Data, Labels):

	"""
	Plot several Raman spectra in individual boxes for comparison.
	"""

# 	_, ColorScheme = colordef(Labels)
	ColorScheme = ['DarkOrange', 'CornflowerBlue', 'OliveDrab']


	xx = Data[:,0].reshape((Data.shape[0],1))
	yy = Data[:,1:]
	nPlots = len(Labels)
	SpecPerPlot = int(yy.shape[1]/nPlots)


	plt.rc('font', family = 'Arial', size='14')


# Subplots with some shared axes
	fig, Axlst = plt.subplots(nPlots, sharex=True, sharey=False, figsize=(6,9))

	for i in range(nPlots):
		for j in range(SpecPerPlot):
			Axlst[i].plot(xx,yy[:,SpecPerPlot*i+j]) #,color=ColorScheme[j])
		Axlst[i].set_yticks(np.arange(0.2, 1.8, 0.2))
		Axlst[i].set_yticklabels([1],visible=False)


# Set different y-limit for colistin due to sulfate peak
		if i == nPlots-1:
			Axlst[i].set_ylim(0,1.7)
		else:
			Axlst[i].set_ylim(0,1.1)
# 		Axlst[i].set_ylim(0,1.1)
		Axlst[i].set_title(Labels[i],ha='left',x=0.05,y=0.8)

	Axlst[-1].set_xlabel(r'Wavenumber ($\mathregular{cm}^{-1}$)')
# 	Axlst[-1].set_ylabel('Normalized Intensity (a.u.)')
	
	fig.text(0.05,0.5,'Normalized Intensity (a.u.)',rotation=90,ha='center',va='center')

	
# Fine-tune figure; remove space btwn subplots and hide x ticks for all but bottom plot
	fig.subplots_adjust(bottom=0.10,left=0.08,right=0.95,top=0.95,hspace=0)
	plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
	
# 	fig.tight_layout()
# 	fig.savefig('test.png')

	plt.show()



#--------------------------------------------------------------------------------------
def specboxdblwide(Data, Labels):

	"""
	Plot several Raman spectra in individual boxes for comparison. 
	The bottom box has an attached box for an additional spectral window.
	"""

	from matplotlib.gridspec import GridSpec
	
# 	_, ColorScheme = colordef(Labels)
	ColorScheme = ['DarkOrange', 'CornflowerBlue', 'OliveDrab']


	xx = Data[:,0].reshape((Data.shape[0],1))
	yy = Data[:,1:]
	nPlots = len(Labels)
	SpecPerPlot = int(yy.shape[1]/nPlots)


# Data for final double-wide spectrum	
	DataEnd = np.loadtxt('AuNPCHIR.csv', delimiter=',', skiprows=1)
	Labels.append('CHIR-090')
	xEnd = DataEnd[:,0].reshape((DataEnd.shape[0],1))
	yEnd = DataEnd[:,1:]

	plt.rc('font', family = 'Arial', size='14')


# Setup grid for all plots and create plots except for last box and plot
	fig = plt.figure(figsize=(7,9))
	gs = GridSpec(nPlots+1,5)
	gs.update(left=0.10, right=0.95, top=0.99, bottom=0.08, hspace=0, wspace=0)
	for i in range(nPlots):
		ax = plt.subplot(gs[i,:-1])
		for j in range(SpecPerPlot):
			ax.plot(xx,yy[:,SpecPerPlot*i+j],c=ColorScheme[j])
		ax.set_yticks(np.arange(0.2, 1.2, 0.2))
		ax.set_yticklabels([1],visible=False)
		ax.set_xticklabels([1],visible=False)	
		ax.set_title(Labels[i],ha='left',x=0.05,y=0.8)
		ax.set_ylim(0,1.1)


# Last box has double-wide plot
	axEndLeft = plt.subplot(gs[-1,:-1])
	axEndRight = plt.subplot(gs[-1,-1])
	for j in range(SpecPerPlot):
		axEndLeft.plot(xEnd,yEnd[:,j],c=ColorScheme[j])
		axEndRight.plot(xEnd,yEnd[:,j]*4,c=ColorScheme[j])
	
	axEndLeft.set_xlim(200,2000)
	axEndLeft.set_yticklabels([1],visible=False)
	axEndLeft.set_title(Labels[-1],ha='left',x=0.05,y=0.8)
	axEndLeft.set_xlabel(r'Wavenumber ($\mathregular{cm}^{-1}$)')
	axEndLeft.set_ylim(0,1.1)

	axEndRight.set_xlim(2000,2400)
	axEndRight.set_xticks([2000,2200,2400])
	axEndRight.set_yticklabels([1],visible=False)
	axEndRight.set_title('(4x)',ha='right',x=0.85,y=0.8)
	axEndRight.set_ylim(0,1.1)
	
	fig.text(0.05,0.5,'Normalized Intensity (a.u.)',rotation=90,ha='center',va='center')

	plt.show()
	
	
	
#--------------------------------------------------------------------------------------
def trenderr(Data, Labels):

	"""
	Plot a scatter plot with errorbars and fit a trendline.
	"""
	
	plt.rc('font', family = 'Arial', size='20')

# 	xx = Data[:,0].reshape((Data.shape[0],1))
	xx = Data[:,0]
	yy = Data[:,1]
	err = Data[:,2]
	
	
# 	Subplots with some shared axes
	fig, Axlst = plt.subplots(1, sharex=True, sharey=False, figsize=(12,7))
	
	
# 	Set margins
	plt.subplots_adjust(bottom=0.10,left=0.1,right=0.95,top=0.95,wspace=0.1,hspace=0.1)
	
	
# 	Scatterplot with error bars
	Axlst.errorbar(xx, yy, fmt='d', yerr=err*2, capsize=5, capthick=1, markersize=9)
	
	
# 	Calculate trendline and plot
	zz = np.polyfit(xx,yy,1)
	p = np.poly1d(zz)
	xtrend = np.linspace(0.8*xx[0],1.01*xx[-1])
	Axlst.plot(xtrend,p(xtrend),'k-')
	
	Axlst.set_xlabel('Concentration (mM)')
	Axlst.set_ylabel('Intensity (a.u.)')
	Axlst.set_xlim(0,21)
	
	plt.show()
	

	
#--------------------------------------------------------------------------------------
def markscatrerr(Data, Labels):

	"""
	Marked scatter plot with error bars.
	"""


	plt.rc('font', family = 'Arial', size='16')
	
	nPlots = len(Labels[1:])
	xx = Data[:,0]
	yy = Data[:,1:]	
	
	fig, axlist = plt.subplots(nPlots, sharex=True, sharey=False, figsize=(6,9))
	for i in range(nPlots):
		axlist[i].errorbar(xx,yy[:,2*i],yerr=yy[:,2*i+1], fmt='kD-', capsize=5, markersize=6, markerfacecolor='black', ecolor='black')
		axlist[i].set_xscale('log')
		axlist[i].set_title(Labels[1+i],ha='left',x=0.05,y=0.8)
	
	
# Customize boxes as necessary for cleaner appearance
	axlist[0].set_ylim(-10, None)
	axlist[2].set_ylim(0, 200)
	
	
# Label bottom figure x-axis
	axlist[-1].set_xlabel(Labels[0])
	axlist[-1].set_xlim(0.5,20000)
	
	
# Label y-axes
	fig.text(0.05,0.5,'Normalized Peak Height',rotation=90,ha='center',va='center')


# Set margins
	plt.subplots_adjust(bottom=0.08,left=0.15,right=0.95,top=0.98,wspace=0.1,hspace=0.1)


	plt.show()



#--------------------------------------------------------------------------------------
def fillbtwn(Data, Labels):

	"""
	Plot min, max, avg y-values and fill between min and max. Data columns: x-values, average, min, max.
	"""
	
	plt.rc('font', family = 'Arial', size='20')

	xx = Data[:,0]
	yy = Data[:,1:]
	
	
# 	Single subplot
	fig, Axlst = plt.subplots(1, sharex=False, sharey=False, figsize=(12,7))
	
	
# 	Set margins
	plt.subplots_adjust(bottom=0.10,left=0.10,right=0.95,top=0.95,wspace=0.1,hspace=0.1)
	
	
# 	Plot average, min, and max
# 	for i in range(yy.shape[1]):
	plt.plot(xx, yy[:,0])
	plt.plot(xx, yy[:,0])
	plt.fill_between(xx,yy[:,1],yy[:,2], where=None, facecolor='gray', edgecolor='black', alpha=0.2)
	plt.plot(xx, yy[:,0], 'k-', linewidth=2)
	
	
	
	Axlst.set_xlabel(r'Wavenumber ($\mathregular{cm}^{-1}$)')
	Axlst.set_ylabel('Intensity (a.u.)')
	Axlst.set_xlim(200,2000)
	
	plt.show()
	
	
	
"""
End function definitions
"""


#--------------------------------------------------------------------------------------
# Main

if __name__ == '__main__':

	# Parse arguments
	parser = argparse.ArgumentParser(description='plot data contained in csv')
	parser.add_argument("pltype", help='options: trenderr, markscatrerr, singlebox, multibox, multiboxdblwide')
	parser.add_argument("datacsv", help='name of csv (including extension)')
	args = parser.parse_args()


	# Check for errors in filename before proceeding to function call
	if re.search('csv', args.datacsv, re.I) is None:
		raise Exception("Requires CSV file with extenstion '.csv'")


	# Open file, read in header, assign labels, close file
	f = open(args.datacsv, mode='r')
	header = f.readline()
	f.close()


	# Raise exception if file is not a CSV
	if ',' not in header:
		raise Exception("Unable to find ',' delimiter, file must contain comma separated values.")


	# Extract data labels from the csv header, first entry is x-axis others are y-axes
	Labels = [s for s in re.split(',',header) if '\n' not in s and s != '']


	# Header is loaded separately as loadtxt does not handle mixed data types well
	Data = np.loadtxt(args.datacsv, delimiter=',', skiprows=1)


	# Call indicated function and report error if not found
	if args.pltype == 'trenderr':
		trenderr(Data, Labels)
	elif args.pltype == 'markscatrerr':
		markscatrerr(Data, Labels)
	elif args.pltype == 'singlebox':
		specsinglebox(Data, Labels)
	elif args.pltype == 'multibox':
		specmultibox(Data, Labels)
	elif args.pltype == 'multiboxdblwide':
		specboxdblwide(Data, Labels)
	elif args.pltype == 'fillbtwn':
		fillbtwn(Data,Labels)
	else:
		raise Exception("Invalid plot type. Use '--help' for options.")


# import ipdb; ipdb.set_trace() # Breakpoint