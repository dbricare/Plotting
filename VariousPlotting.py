"""
Module for creating various matplotlib graphs
"""



"""
Begin function definitions
"""

#--------------------------------------------------------------------------------------
# Plot a scatter plot with errorbars and fit a trendline


def trenderr(Data, Labels):
	
	plt.rc('font', family = 'Arial', size='20')

# 	xx = Data[:,0].reshape((Data.shape[0],1))
	xx = Data[:,0]
	yy = Data[:,1]
	err = Data[:,2]
	
# 	import ipdb; ipdb.set_trace() # Breakpoint
	
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
# Marked scatter plot with error bars

def markscatrerr(Data, Labels):

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
# Parse the arguments passed by the user and run the selected plotting function


import argparse, re
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

# Parse arguments
parser = argparse.ArgumentParser(description='plot data contained in csv')
parser.add_argument("pltype", help='plot options: markscatrerr, trenderr')
parser.add_argument("datacsv", help='name of csv containing data to plot')


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
# Remove the first label for the x-axis data
# Labels = Labels[1:]


# Header is loaded separately as loadtxt does not handle mixed data types well
Data = np.loadtxt(args.datacsv, delimiter=',', skiprows=1)


# Call indicated function and report error if not found
if args.pltype == 'trenderr':
	trenderr(Data, Labels)
if args.pltype == 'markscatrerr':
	markscatrerr(Data, Labels)
else:
	raise Exception("Invalid plot type. Use '--help' for options.")



# import ipdb; ipdb.set_trace() # Breakpoint