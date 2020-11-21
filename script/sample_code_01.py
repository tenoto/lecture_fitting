#!/usr/bin/env python

import os 
import numpy as np
import pandas as pd
import matplotlib.pylab as plt 

from iminuit import Minuit
from probfit import BinnedChi2, Chi2Regression, Extended, gaussian, linear 

df_137cs = pd.read_csv('data/201109_id30_137cs_15min.csv',
	index_col=False, names=['sec','pha'],
	dtype={'sec':np.float,'pha':np.uint16})
print(df_137cs)

df_bgd = pd.read_csv('data/201109_id30_bkg_15min.csv',
	index_col=False, names=['sec','pha'],
	dtype={'sec':np.float,'pha':np.uint16})
print(df_bgd)

##### 

os.system('rm -rf out; mkdir -p out;')

##### 

nbins = 2**9 
tmax = 2**9 

fig = plt.figure(figsize=(8,5)) 
plt.hist(df_137cs['sec'],range=(0,tmax),bins=nbins,histtype='step')
plt.hist(df_bgd['sec'],range=(0,tmax),bins=nbins,histtype='step')
plt.xlim(0.0,tmax)
plt.xlabel('Time (sec)', fontsize=10)
plt.ylabel('Counts / 1-sec bin', fontsize=10)
plt.tight_layout()
plt.tick_params(axis="x",direction="in")
plt.tick_params(axis="y",direction="in")
plt.savefig('out/lc_137cs_bgd_1sec_hist.png')


##### 

lc_src_y, edge = np.histogram(df_137cs['sec'],range=(0,tmax),bins=nbins)
lc_src_x = (edge[:-1] + edge[1:]) / 2.

lc_bgd_y, edge = np.histogram(df_bgd['sec'],range=(0,tmax),bins=nbins)
lc_bgd_x = (edge[:-1] + edge[1:]) / 2.

fig = plt.figure(figsize=(8,5)) 
plt.errorbar(lc_src_x,lc_src_y,yerr=np.sqrt(lc_src_y),marker='',drawstyle='steps-mid')
plt.errorbar(lc_bgd_x,lc_bgd_y,yerr=np.sqrt(lc_bgd_y),marker='',drawstyle='steps-mid')
plt.xlim(0.0,tmax)
plt.xlabel('Time (sec)', fontsize=10)
plt.ylabel('Counts / 1-sec bin', fontsize=10)
plt.tight_layout()
plt.tick_params(axis="x",direction="in")
plt.tick_params(axis="y",direction="in")
plt.savefig('out/lc_137cs_bgd_1sec.png')


#####

tbin = 2**3 # 8 sec
nbins = int(nbins / tbin)

lc_src_y, edge = np.histogram(df_137cs['sec'],range=(0,tmax),bins=nbins)
lc_src_x = (edge[:-1] + edge[1:]) / 2.

lc_bgd_y, edge = np.histogram(df_bgd['sec'],range=(0,tmax),bins=nbins)
lc_bgd_x = (edge[:-1] + edge[1:]) / 2.

fig = plt.figure(figsize=(8,5)) 
plt.errorbar(lc_src_x,lc_src_y,yerr=np.sqrt(lc_src_y),marker='',drawstyle='steps-mid')
plt.errorbar(lc_bgd_x,lc_bgd_y,yerr=np.sqrt(lc_bgd_y),marker='',drawstyle='steps-mid')
plt.xlim(0.0,tmax)
plt.xlabel('Time (sec)', fontsize=10)
plt.ylabel('Counts / (%d-sec bin)' % tbin, fontsize=10)
plt.tight_layout()
plt.tick_params(axis="x",direction="in")
plt.tick_params(axis="y",direction="in")
plt.savefig('out/lc_137cs_bgd_%dsec.png' % tbin)

fig = plt.figure(figsize=(8,5)) 
plt.errorbar(lc_src_x,lc_src_y/float(tbin),yerr=np.sqrt(lc_src_y)/float(tbin),marker='',drawstyle='steps-mid')
plt.errorbar(lc_bgd_x,lc_bgd_y/float(tbin),yerr=np.sqrt(lc_bgd_y)/float(tbin),marker='',drawstyle='steps-mid')
plt.xlim(0.0,tmax)
plt.xlabel('Time (sec)', fontsize=10)
plt.ylabel('Counts / sec (%d-sec bin)' % tbin, fontsize=10)
plt.tight_layout()
plt.tick_params(axis="x",direction="in")
plt.tick_params(axis="y",direction="in")
plt.savefig('out/lc_137cs_bgd_%dsec_scale.png' % tbin)

##### 

lc_src_y, edge = np.histogram(df_137cs['sec'],range=(0,tmax),bins=tmax)

xmin = 300
xwid = 2**8
xmax = xmin + xwid 
nbins = 2**6

hist_y, hist_edge = np.histogram(lc_src_y,range=(xmin,xmax),bins=nbins)
hist_x = (hist_edge[:-1] + hist_edge[1:]) / 2.

fig = plt.figure(figsize=(8,5)) 
plt.errorbar(hist_x,hist_y,yerr=np.sqrt(hist_y),marker='',drawstyle='steps-mid')
plt.xlim(xmin,xmax)
plt.xlabel('Rate (counts/sec)', fontsize=10)
plt.ylabel('Number of bins', fontsize=10)
plt.tight_layout()
plt.tick_params(axis="x",direction="in")
plt.tick_params(axis="y",direction="in")
plt.savefig('out/hist_lc_137cs_1sec.png')

print(np.mean(lc_src_y))


def mygauss(x, mu, sigma, area):
	return area * np.exp(-0.5*(x-mu)**2/sigma**2)/(np.sqrt(2*np.pi)*sigma)

fig = plt.figure(figsize=(8,5)) 
plt.errorbar(hist_x,hist_y,yerr=np.sqrt(hist_y),marker='',drawstyle='steps-mid')
plt.plot(hist_x,[mygauss(x,430,20,1000) for x in hist_x])
plt.xlim(xmin,xmax)
plt.xlabel('Rate (counts/sec)', fontsize=10)
plt.ylabel('Number of bins', fontsize=10)
plt.tight_layout()
plt.tick_params(axis="x",direction="in")
plt.tick_params(axis="y",direction="in")
plt.savefig('out/hist_lc_137cs_1sec_tmp1.png')


chi2reg = Chi2Regression(mygauss,hist_x,hist_y)
fit = Minuit(chi2reg,mu=430,sigma=20,area=1000)
fit.migrad()
fit.minos() 
print(fit.print_param())
print(fit.values)
print(fit.errors)

fig = plt.figure(figsize=(8,5)) 
plt.errorbar(hist_x,hist_y,yerr=np.sqrt(hist_y),marker='',drawstyle='steps-mid')
model_mygauss = [mygauss(x,fit.values[0],fit.values[1],fit.values[2]) for x in hist_x]
plt.plot(hist_x,model_mygauss)
plt.xlim(xmin,xmax)
plt.xlabel('Rate (counts/sec)', fontsize=10)
plt.ylabel('Number of bins', fontsize=10)
plt.tight_layout()
plt.tick_params(axis="x",direction="in")
plt.tick_params(axis="y",direction="in")
plt.savefig('out/hist_lc_137cs_1sec_fit.png')

print(np.mean(lc_src_y))
print(fit.values[0])
print(fit.values[1])
print(np.sqrt(fit.values[0]))

##### 

print(np.diff(df_137cs['sec']))

xmin = 0.0005 
xmax = 0.02 + 0.0005 
nbins = 20

deltat_y, deltat_edge = np.histogram(np.diff(df_137cs['sec']),range=(xmin,xmax),bins=nbins)
deltat_x = (deltat_edge[:-1] + deltat_edge[1:]) / 2.

fig = plt.figure(figsize=(8,5)) 
plt.errorbar(1000.0*deltat_x,deltat_y,
	yerr=np.sqrt(deltat_y),marker='',drawstyle='steps-mid')
#plt.xlim(0.0,900.0)
plt.xlabel('Delta time (msec)', fontsize=10)
plt.ylabel('Events', fontsize=10)
plt.xscale('linear')				
plt.yscale('log')
plt.tight_layout()
plt.tick_params(axis="x",direction="in")
plt.tick_params(axis="y",direction="in")

def myexp(x, mu, norm):
	return norm * np.exp(-x/mu)

chi2reg = Chi2Regression(myexp,deltat_x,deltat_y)
fit = Minuit(chi2reg, mu=0.002356, norm=0.002356)
fit.migrad()
fit.minos() 
print(fit.print_param())
print(fit.values)
print(fit.errors)

model_myexp = [myexp(x,fit.values[0],fit.values[1]) for x in deltat_x]
plt.plot(1000*deltat_x,model_myexp,c='red')
plt.savefig('out/hist_deltat_137cs_fit.png')


##### 

exposure = 50 # sec

flag_137cs = df_137cs['sec'] < exposure 
flag_bgd = df_bgd['sec'] < exposure 

spec_137cs_y, edge = np.histogram(df_137cs['pha'][flag_137cs], 
	bins=2**10, range=(-0.5, 2**10-0.5))
spec_137cs_x = (edge[:-1] + edge[1:]) / 2.
spec_137cs_x_err = (edge[:-1] - edge[1:]) / 2.
spec_137cs_y_err = np.sqrt(spec_137cs_y)

spec_bgd_y, edge = np.histogram(df_bgd['pha'][flag_bgd], 
	bins=2**10, range=(-0.5, 2**10-0.5))
spec_bgd_x = (edge[:-1] + edge[1:]) / 2.
spec_bgd_x_err = (edge[:-1] - edge[1:]) / 2.
spec_bgd_y_err = np.sqrt(spec_bgd_y)

fig, ax = plt.subplots(1,1, figsize=(11.69,8.27))
plt.errorbar(spec_137cs_x,spec_137cs_y,yerr=spec_137cs_y_err,
	marker='',drawstyle='steps-mid')
plt.errorbar(spec_bgd_x,spec_bgd_y,yerr=spec_bgd_y_err,
	marker='',drawstyle='steps-mid')
#plt.errorbar(x_bgd,y_bgd,yerr=np.sqrt(y_bgd),marker='',drawstyle='steps-mid')
plt.xlabel('ADC channel (pha)', fontsize=10)
plt.ylabel('Counts', fontsize=10)
#plt.xscale('log')				
plt.yscale('log')
plt.xlim(-0.5,300-0.5)
plt.tight_layout(pad=2)
#plt.tick_params(labelsize=fontsize)
#plt.rcParams["font.family"] = "serif"
#plt.rcParams["mathtext.fontset"] = "dejavuserif"	
plt.savefig('out/spec_137cs_bgd.png')

##### 

#exposure = 15.0 * 60 # sec
spec_137cs_sub_y = (spec_137cs_y - spec_bgd_y) / exposure 
spec_137cs_sub_yerr = np.sqrt(spec_137cs_y + spec_bgd_y) / exposure
spec_137cs_sub_x = spec_137cs_x
spec_137cs_sub_xerr = spec_137cs_x_err

chmin = 40
chmax = 2**6 

x = spec_137cs_sub_x[chmin:chmax]
y = spec_137cs_sub_y[chmin:chmax]
xe = spec_137cs_sub_xerr[chmin:chmax]
ye = spec_137cs_sub_yerr[chmin:chmax]

fig, ax = plt.subplots(1,1, figsize=(11.69,8.27))
plt.errorbar(x,y,yerr=ye,xerr=xe,marker='',drawstyle='steps-mid')
plt.xlabel('ADC channel (pha)', fontsize=10)
plt.ylabel('Counts', fontsize=10)
#plt.xscale('log')				
plt.yscale('log')
plt.xlim(-0.5,300-0.5)
#plt.tight_layout(pad=2)
#plt.tick_params(labelsize=fontsize)
#plt.rcParams["font.family"] = "serif"
#plt.rcParams["mathtext.fontset"] = "dejavuserif"	

def mygauss_linear(x, mu, sigma, area, c0=0.0, c1=0.0):
    return area * np.exp(-0.5*(x-mu)**2/sigma**2)/(np.sqrt(2*np.pi)*sigma) + c0 + c1 * x

chi2reg = Chi2Regression(mygauss_linear,x,y)
fit = Minuit(chi2reg, mu=50, sigma=2, area=300, c0=0.0, c1=0.0)
fit.migrad()
fit.minos() 
print(fit.print_param())
print(fit.values)
print(fit.errors)


fig = plt.figure(figsize=(7,6))
ax1 = fig.add_axes([0.1, 0.3, 0.85, 0.68])
ax1.errorbar(x,y,xerr=xe,yerr=ye,marker='o',ls='',color='k')
model_mygauss_linear = [mygauss_linear(i,
	fit.values[0],fit.values[1],fit.values[2],fit.values[3],fit.values[4]) for i in x]
#model_mygauss_linear = [mygauss_linear(i,
#	50,2,100,0,0) for i in x]
ax1.plot(x,model_mygauss_linear,c='red',drawstyle='steps-mid')
ax1.set_xlim(chmin,chmax)
#plt.xlabel('ADC channel', fontsize=10)
ax1.set_ylabel('Rate (counts/sec/bin)', fontsize=10)
ax1.axhline(y=0.0, color='k', linestyle='--')
ax1.get_xaxis().set_visible(False)
plt.tight_layout()
plt.tick_params(axis="x",direction="in")
plt.tick_params(axis="y",direction="in")
# Second new axes

y_residual = y - model_mygauss_linear
ax2 = fig.add_axes([0.1, 0.1, 0.85, 0.20])
ax2.errorbar(x,y_residual,xerr=xe,yerr=ye,marker='o',ls='',color='k')
ax2.set_xlim(chmin,chmax)
ax2.axhline(y=0.0, color='r', linestyle='-')
ax2.set_ylim(-3.0,3.0)
ax2.set_xlabel('ADC channel', fontsize=10)
ax2.set_ylabel('Residual', fontsize=10)
#fig.align_ylabels()
plt.tick_params(axis="x",direction="in")
plt.tick_params(axis="y",direction="in")
plt.savefig('out/spec_137cs_bgd_fit.png')

print(fit.values[2])

#####

ch_662keV_mu = fit.values[0]
ch_662keV_sigma = fit.values[1]
print(ch_662keV_mu,ch_662keV_sigma)
chmin_662keV = ch_662keV_mu - 3.0 * ch_662keV_sigma
chmax_662keV = ch_662keV_mu + 3.0 * ch_662keV_sigma
print(chmin_662keV,chmax_662keV)

flag_137cs_662keV = np.logical_and(df_137cs["pha"] >= chmin_662keV, df_137cs["pha"] < chmax_662keV)
flag_bgd_662keV = np.logical_and(df_bgd["pha"] >= chmin_662keV, df_bgd["pha"] < chmax_662keV)

lc_137cs_y, edge = np.histogram(df_137cs['sec'][flag_137cs_662keV],range=(0,900.),bins=900)
lc_137cs_x = (edge[:-1] + edge[1:]) / 2.

bgd_lc_y, edge = np.histogram(df_bgd['sec'][flag_bgd_662keV],range=(0,900.),bins=900)
bgd_lc_x = (edge[:-1] + edge[1:]) / 2.

lc_137cs_mean = np.mean(lc_137cs_y)
lc_bgd_mean = np.mean(bgd_lc_y)

fig = plt.figure(figsize=(8,5)) 
plt.errorbar(lc_137cs_x,lc_137cs_y-lc_bgd_mean,yerr=np.sqrt(lc_137cs_y),marker='',drawstyle='steps-mid')
#plt.errorbar(bgd_lc_time,bgd_lc_cnt,yerr=np.sqrt(bgd_lc_cnt),marker='',drawstyle='steps-mid')
plt.xlim(0.0,100.0)
plt.xlabel('Time (sec)', fontsize=10)
plt.ylabel('Counts / 1-sec bin', fontsize=10)
plt.tight_layout()
plt.tick_params(axis="x",direction="in")
plt.tick_params(axis="y",direction="in")
plt.savefig('out/lc_137cs_bgd_energy_cut.png')

print(lc_137cs_mean)
print(lc_bgd_mean)
print(lc_137cs_mean-lc_bgd_mean)

####

df_60co = pd.read_csv('data/201119_id30_60co_15min.csv',
	index_col=False, names=['sec','pha'],
	dtype={'sec':np.float,'pha':np.uint16})
print(df_60co)

exposure = 50

flag_time = df_60co['sec'] < exposure 

y_src, edges_src = np.histogram(df_60co['pha'][flag_time], bins=2**10, range=(-0.5, 2**10-0.5))
x_src = (edges_src[:-1] + edges_src[1:]) / 2.
x_err = (edges_src[:-1] - edges_src[1:]) / 2.

y_bgd, edges_bgd = np.histogram(df_bgd['pha'][flag_bgd], bins=2**10, range=(-0.5, 2**10-0.5))
x_bgd = (edges_bgd[:-1] + edges_bgd[1:]) / 2.

fig, ax = plt.subplots(1,1, figsize=(11.69,8.27))
plt.errorbar(x_src,y_src,yerr=np.sqrt(y_src),marker='',drawstyle='steps-mid')
plt.errorbar(x_bgd,y_bgd,yerr=np.sqrt(y_bgd),marker='',drawstyle='steps-mid')
plt.xlabel('ADC channel (pha)', fontsize=10)
plt.ylabel('Counts', fontsize=10)
plt.yscale('log')
plt.xlim(-0.5,300-0.5)
plt.savefig('out/spec_60co_bgd.png')


y_sub = (y_src - y_bgd) / exposure 
y_sub_error = np.sqrt(y_src + y_bgd) / exposure
x_sub = x_src
x_sub_error = x_err

chmin = 75
chmax = 115

x_sub_fit = x_sub[chmin:chmax]
y_sub_fit = y_sub[chmin:chmax]
y_sub_error_fit = y_sub_error[chmin:chmax]
x_sub_error_fit = x_sub_error[chmin:chmax]

fig, ax = plt.subplots(1,1, figsize=(11.69,8.27))
plt.errorbar(x_sub_fit,y_sub_fit,yerr=y_sub_error_fit,marker='',drawstyle='steps-mid')
plt.xlabel('ADC channel (pha)', fontsize=10)
plt.ylabel('Counts', fontsize=10)
plt.yscale('log')
#plt.xlim(-0.5,300-0.5)
plt.savefig('out/spec_60co_bgd_sub.png')

def my2gauss_linear(x, mu1, sigma1, area1, mu2, sigma2, area2, c0=0.0, c1=0.0,c2=0.0):
    return area1 * np.exp(-0.5*(x-mu1)**2/sigma1**2)/(np.sqrt(2*np.pi)*sigma1) + area2 * np.exp(-0.5*(x-mu2)**2/sigma2**2)/(np.sqrt(2*np.pi)*sigma2) + c0 + c1 * x + c2 * x * x

chi2reg = Chi2Regression(my2gauss_linear,x_sub_fit,y_sub_fit)
fit = Minuit(chi2reg, mu1=80, sigma1=2, area1=300, mu2=100, sigma2=2, area2=300, c0=0.0, c1=0.0, c2=0.0)
fit.migrad()
fit.minos() 
print(fit.print_param())
print(fit.values)
print(fit.errors)

fig = plt.figure(figsize=(7,6))
ax1 = fig.add_axes([0.1, 0.3, 0.85, 0.68])
ax1.errorbar(x_sub_fit,y_sub_fit,xerr=x_sub_error_fit,yerr=y_sub_error_fit,marker='o',ls='',color='k')
model_my2gauss_linear = [my2gauss_linear(i,
	fit.values[0],fit.values[1],fit.values[2],
	fit.values[3],fit.values[4],fit.values[5],
	fit.values[6],fit.values[7],fit.values[8]
	) for i in x_sub_fit]
#model_my2gauss_linear = [my2gauss_linear(i,
#	80,2,300,100,2,300,0.0,0.0
#	) for i in x_sub_fit]
ax1.plot(x_sub_fit,model_my2gauss_linear,c='red',drawstyle='steps-mid')
ax1.set_xlim(chmin,chmax)
#plt.xlabel('ADC channel', fontsize=10)
ax1.set_ylabel('Rate (counts/sec/bin)', fontsize=10)
ax1.axhline(y=0.0, color='k', linestyle='--')
ax1.get_xaxis().set_visible(False)
plt.tight_layout()
plt.tick_params(axis="x",direction="in")
plt.tick_params(axis="y",direction="in")
# Second new axes

y_residual = y_sub_fit - model_my2gauss_linear
ax2 = fig.add_axes([0.1, 0.1, 0.85, 0.20])
ax2.errorbar(x_sub_fit,y_residual,xerr=x_sub_error_fit,yerr=y_sub_error_fit,marker='o',ls='',color='k')
ax2.set_xlim(chmin,chmax)
ax2.axhline(y=0.0, color='r', linestyle='-')
ax2.set_ylim(-3.0,3.0)
ax2.set_xlabel('ADC channel', fontsize=10)
ax2.set_ylabel('Residual', fontsize=10)
#fig.align_ylabels()
plt.tick_params(axis="x",direction="in")
plt.tick_params(axis="y",direction="in")
plt.savefig('out/spec_60co_bgd_fit.png')


####

x = np.array([661.65,1173.23,1332.50],dtype='float') # energy 
y = np.array([50.46,89.90,102.27],dtype='float') # channel 
ye = np.array([0.08,0.05,0.06],dtype='float')

x2r = Chi2Regression(linear, x, y, ye)
fit = Minuit(x2r, m=1, c=2)
fit.migrad()
fit.minos() 
print(fit.print_param())
print(fit.values)
print(fit.errors)

fig = plt.figure(figsize=(8,5)) 
plt.errorbar(x,y,yerr=ye,marker='o',ls='',color='k')
model_x = range(0,15000)
model_linear = [linear(i,fit.values[0],fit.values[1]) for i in model_x]
plt.plot(model_x,model_linear,color='r',ls='--')
plt.xlim(0,15000)
plt.ylim(0,2**10)
plt.xlabel('Energy (keV)', fontsize=10)
plt.ylabel('ADC channel', fontsize=10)
plt.tight_layout()
plt.tick_params(axis="x",direction="in")
plt.tick_params(axis="y",direction="in")
plt.savefig('out/energy_vs_channel.png')


### 

exposure = 15 * 60  

hist_y, edge = np.histogram(df_bgd['pha'], bins=2**10,range=(-0.5, 2**10-0.5))
hist_x = (edge[:-1] + edge[1:]) / 2.
hist_xe = (edge[:-1] - edge[1:]) / 2.
hist_ye = np.sqrt(hist_y)

x = (hist_x+0.644)/0.0772/1000.0
xe = hist_xe/0.0772/1000.0
y = 1/77.2 * hist_y / exposure 
ye = 1/77.2 * hist_ye / exposure 

fig, ax = plt.subplots(1,1,figsize=(11.69,8.27))
plt.errorbar(x,y,xerr=xe,yerr=ye,marker='',drawstyle='steps-mid',color='k')
plt.xlabel('Energy (MeV)', fontsize=13)
plt.ylabel('Counts/sec/keV', fontsize=13)
plt.xscale('log')				
plt.yscale('log')
plt.xlim(0.3,13.0)
plt.tight_layout(pad=2)
plt.tick_params(labelsize=12)
plt.tick_params(axis="x",direction="in")
plt.tick_params(axis="y",direction="in")
plt.savefig('out/spec_bgd_energy.png')

exit()

"""
#### 

emin = 0.3
emax = 15.0 

hist_y, edge = np.histogram((df_bgd['pha']+0.644)/0.0772/1000., 
	bins=10**np.linspace(np.log10(emin), np.log10(emax), 2**8))
hist_x = (edge[:-1] + edge[1:]) / 2.
hist_xe = (edge[:-1] - edge[1:]) / 2.
hist_ye = np.sqrt(hist_y)

print(hist_ye)

fig, ax = plt.subplots(1,1, figsize=(11.69,8.27))
plt.errorbar(hist_x,hist_y,yerr=hist_ye,marker='',drawstyle='steps-mid')
plt.xlabel('Energy (MeV)', fontsize=10)
plt.ylabel('Counts', fontsize=10)
plt.gca().set_xscale("log")
#plt.xscale('log')				
plt.yscale('log')
#plt.xlim(-0.5,300-0.5)
plt.tight_layout(pad=2)
#plt.tick_params(labelsize=fontsize)
#plt.rcParams["font.family"] = "serif"
#plt.rcParams["mathtext.fontset"] = "dejavuserif"	
plt.savefig('out/spec_bgd_energy.png')
"""

