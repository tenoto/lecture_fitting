#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pylab as plt 
from probfit import Chi2Regression
from iminuit import Minuit

def polfit(x, c0=0.0,c1=1.0,p=-1.0):
	return c0+c1*x**p

df = pd.read_csv('data/count_600sec.csv',
	dtype={'distance(m)':float,'phaall':int,'pha>1MeV':int,'pha>200KeV':int})
# print(df)

exposure = 600.0
data_x = np.array(df['distance(m)'])
data_y = np.array(df['pha>200KeV']/exposure)
data_yerr = np.array(np.sqrt(df['pha>200KeV'])/exposure)

model_x = np.linspace(0.1,5.0,100)

fig = plt.figure(figsize=(8,5)) 
plt.errorbar(data_x,data_y,yerr=data_yerr,fmt='o',label='data')
plt.plot(model_x,polfit(model_x,c0=12.0,c1=40.0,p=-1.4),label='initial guess')
plt.xlim(0.3,6.0)
plt.ylim(10,200)
plt.yscale('log')
plt.xscale('log')

plt.grid(True)
plt.xlabel('Distance (m)', fontsize=10)
plt.ylabel('Count rate (cps)', fontsize=10)
plt.tight_layout()
plt.tick_params(axis="x",direction="in")
plt.tick_params(axis="y",direction="in")
plt.savefig('fit.pdf')

chi2reg = Chi2Regression(polfit,data_x,data_y)
fit = Minuit(chi2reg,c0=12.0,c1=40.0,p=-1.4)
fit.migrad()
fit.minos() 
print(fit.print_param())
print(fit.values)
print(fit.errors)

fit_c0 = fit.values[0]
fit_c1 = fit.values[1]
fit_p = fit.values[2]

plt.plot(model_x,polfit(model_x,c0=fit_c0,c1=fit_c1,p=fit_p),label='fit')
plt.legend()
plt.savefig('fit.pdf')