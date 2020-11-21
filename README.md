# Python による放射線データの解析

[tenoto/lecture_fitting](https://github.com/tenoto/lecture_fitting) @ GitHub

###### tags: `lecture` `python`　[![hackmd-github-sync-badge](https://hackmd.io/H__Gt0oGQGub8l8cusKdww/badge)](https://hackmd.io/H__Gt0oGQGub8l8cusKdww)


高エネルギー天文学で使われる HEASoft や素粒子・原子核実験で使われる ROOT などの専用解析ツールを使わず、データ解析を行いたい場合もあります。Python だけで放射線データの基本的な解析をする手順を記載します。 

## 練習: 乱数生成した標準正規分布をフィット

まず、実際の測定データを扱う前に、疑似データを乱数から生成して遊んでみましょう。

この一連のページでは、Python での解析と実例を紹介します。コードは [Google Colaboratory](https://colab.research.google.com/drive/1pCiQg6YyaCpmp3GhzG79j9OS1wcFFGPK?usp=sharing) に一部置いています。Google Colaboratory では標準で多くのライブラリをもっていますが、足りないものは `!pip install xxx` で使えるようにできます。

以下のすべてを Google Colab には置いていないので、注意してください。後半では、GitHub にコードを置いているで、Google Colab でも GitHub でも使いやすい方を試してください

### 乱数

乱数はいくつかのライブラリで扱えますが、様々な目的で使える numpy で呼ぶのが使いやすいことが多いです。

```python
import numpy as np
np.random.seed(0)
evt = np.random.randn(1000)
```

ここで、`seed(0)` は乱数のシード設定、`randn(1000)` は標準正規分布(平均 0, 標準偏差 1)に従う乱数を 1000 個、生成して numpy の array に収容しています。この array は、素粒子物理学やＸ線天文学などで、放射線イベントの何らかの物理量の列に対応する、というイメージで evt という変数名にしてあります。

### ヒストグラムに詰める

生成された数値データをヒストグラムに詰めてプロットするには、`numpy.histogram` が使えます。ヒストグラムの上限、下限とその間のビン数を指定するのは、ROOT (もっと古い古代語だと display45, pow?)などと同じにできます。

```python
hy, xedges = np.histogram(evt,bins=40,range=(-4.,4.))
hx = 0.5*(xedges[1:]+xedges[:-1]) 
hyerr = np.sqrt(hy)
```

hy はヒストグラムに詰められた値で、xedges はヒストグラムの境界の値のため、ヒストグラムの中心値は `hx = 0.5*(xedges[1:]+xedges[:-1])` で計算して hx に詰めておく必要があります。統計誤差は、各ヒストグラムの平方根で hyerr に詰めています。

:::warning
(注意) 実は、天文学で扱うことがある小統計の統計誤差について、平方根を使うのには注意が必要です。[Gehrels "N.Confidence Limits for Small Numbers of Events in Astrophysical Data", ApJ, 303, 336 (1986)](https://ui.adsabs.harvard.edu/abs/1986ApJ...303..336G/abstract) や、HEASoft の mathpha のコード中での取り扱い [Ian "The MATHPHA User’s Guide" (1995)](https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/ogip_95_008/ogip_95_008.pdf)　の "3 CALCULATION & PROGATION OF ERRORS" などが参考になります。
:::

これらを用いて、
```
fig = plt.figure(figsize=(6,4)) 
plt.errorbar(hx, hy, yerr=hyerr, marker='', drawstyle='steps-mid') 
```
とすれば、

![](https://i.imgur.com/WJKzPow.png)

のようにプロットできます。`marker='', drawstyle='steps-mid'` のオプションにすることで、標準的な物理での誤差付きヒストグラムの表現にしています。

### フィッティング

上のヒストグラムは、標準正規分布から生成しているので、十分に統計を溜めれば、平均 0 で、標準偏差 1 のフィット結果に収束するはずです。

素粒子物理学で蓄積されてきたフィット（より正確には、評価関数の最小化）のアルゴリズムは、Minuit のライブラリになっており、python では、iminuit という名前で、一般的な関数最小化法として使用できます。これは、モデルのデータへの尤度フィットや、尤度プロファイル分析から、モデルのパラメータ誤差推定値を得るためにも使用できます。[iminuit チュートリアル](https://nbviewer.jupyter.org/github/scikit-hep/iminuit/tree/master/tutorial/)が参考になります。

さらに、probfit で iminuit で使いたい標準的な関数や、尤度関数、Chi-square 計算などを呼べるようにします。

#### Chi-square フィッティング

```python
from iminuit import Minuit
from probfit import BinnedChi2, Extended, gaussian

ext_gauss = Extended(gaussian)
bc2 = BinnedChi2(ext_gauss, evt)

m = Minuit(bc2, mean=-1, sigma=0.5, N=100)

plt.figure(figsize=(12, 4))
plt.subplot(121)
bc2.draw(m)
plt.title('Before')

m.migrad() # fit

plt.subplot(122)
bc2.draw(m)
plt.title('After')
```

gaussian 関数を、Extended することで規格化因子を付けています。その上で、BinnedChi2 でガウス関数を生成したデータに適用したときの評価関数をつくり、Minuit で最小化できるように準備します。この際、初期値を入れています。`m.migrad()`がこの評価関数の最小値を探してくる本体です。

最小値を求めた後、以下のコマンドで非対称な誤差を計算します。
```
m.minos() 
```
![](https://i.imgur.com/oJE6qFd.png)

これらの誤差は、以下で表示できます。
```
m.print_param()
```
![](https://i.imgur.com/Z6Gu8ye.png)

フィットした後のパラメータや誤差を取り出すには、
```
m.values
m.errors
m.merrors
```
などとすればよいです。上記のフィットの結果から、中心値は $-0.046\pm 0.032$ なので、$2\sigma$の範囲で最初の乱数生成の仮定と一致し、幅については $0.987\pm 0.026$ と$1\sigma$の範囲で一致していることがわかります。

なお、今回は `BinnedChi2` を使いましたが、他にも色々あります（詳細は別途）。


## 測定セットアップ

では、ここから実際に、測定データで試してみます。

- 放射線測定は、2020年11月9日、19日に、理化学研究所の榎戸研実験室(研究本館 610号室)で、茨城高専の村上悦基君、理研・榎戸研の沼澤正樹さんにやっていただきました。
- 放射線源として、ここでは 137Cs, 60Co の 2つの測定データと、放射線源を置かない場合の環境放射線バックグラウンドの測定データを扱う。これらの線源から放出される代表的なガンマ線輝線のエネルギーは[物理定数表](https://github.com/tenoto/repository/blob/master/docs/physical_const_entv190819j.pdf)にも記載している。
- 検出器は「雷雲プロジェクト」用に開発してきた、コガモ検出器(Compact Gamma-ray Monitor; CoGaMo)を用いた。5x5x15 cm の CsI(Tl) 結晶シンチレータと MPPC 光検出器を用いている。

![](https://i.imgur.com/W9NIwlX.jpg)

## 測定データと解析スクリプト

- 解析スクリプトは、[tenoto/lecture_fitting@GitHub](https://github.com/tenoto/lecture_fitting) からダウンロードできる。
- 測定データは(今回の実習用に加工したものを) [Dropbox](https://www.dropbox.com/s/1birsl4472v2ipg/data.zip?dl=0) からダウンロードでき、GitHub からプルした lecture_fitting の直下に置いてください。

### 準備

まず、必要なライブラリを読み込んでおきます。
```
#!/usr/bin/env python

import os 
import numpy as np
import pandas as pd
import matplotlib.pylab as plt 

from iminuit import Minuit
from probfit import BinnedChi2, Chi2Regression, Extended, gaussian
```

最後の iminuit と probfit が今回の放射線計測のために重要で、これらのライブラリは、`pip install xxx` (xxx はライブラリ名)などで準備してください。

### CSV 形式のデータ読み込み

data ディレクトリ下に置いてある`data/201109_id30_137cs_15min.csv` という CSV ファイルは、Time (sec) と ADC channel (pha) の列になっています。

python のライブラリ pandas には csv 形式を読める read_csv が用意されています。詳しくは、仕様を読んでもらえれば良いですが、コラム名はファイル内には含まれていないので、`index_col=False, names=['sec','pha']`で指定しています。

また、`dtype` でコラムのフォーマットを指定しています。`sec` の方は、誤ったフォーマットにすると数値がずれてしまうので注意。

読み込んだものは、data frame (df) に取り込まれます。

```
df_137cs = pd.read_csv('data/201109_id30_137cs_15min.csv',
	index_col=False, names=['sec','pha'],
	dtype={'sec':np.float,'pha':np.uint16})
print(df_137cs)

df_bgd = pd.read_csv('data/201109_id30_bkg_15min.csv',
	index_col=False, names=['sec','pha'],
	dtype={'sec':np.float,'pha':np.uint16})
print(df_bgd)
```

## ライトカーブ（ヒストグラムを表示する）

### matplotlib を使う方法

```
os.system('rm -rf out; mkdir -p out;')

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
plt.savefig('out/lc_137cs_bgd_1sec_hist.gif')
```

matplotlib の hist を使って、ヒストグラムを描画できます。ヒストグラムは上限、下限とビン数を指定し、表示方法を指定できます。今回、ビン数は 2^9 にしています。これは、ビンまとめするときに、2の乗数だと便利なことが多いからです。別に整数なら何でも構いません。

![](https://i.imgur.com/RE8zmH7.png)

この図で、上の青は 137Cs、下のオレンジは環境放射線（何も放射線源を当てていない）の場合です。放射線を当てた方が、150 cps ほど増えていることがわかります。

### numpy を使う方法

```
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
```

同じことを numpy でもできます。この際、yerr で誤差をつけてみました。

![](https://i.imgur.com/QnHt6Pm.png)

### ビンの調整

今回の測定では、1秒でもそれなりの統計がありますが、例えば 8 秒でライトカーブが描きたい場合は以下のようになります。

```
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
```

![](https://i.imgur.com/YBMvJb9.png)

最初のプロットでは、8 秒ビンごとの個数が表示されているので、1秒あたりに換算しています

![](https://i.imgur.com/7LGMHPU.png)

## ライトカーブを解析する

### ライトカーブのヒストグラムを表示する

今回の測定は時間的に一定で、平均値は 426.5 counts/s です。`np.mean(lc_src_y)` とかやると計算できます。では、１秒値のライトカーブのレートがどの程度、分散があるのか調べてみます。これは、1秒値のレートの array をヒストグラムに詰めることになります。

```
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
```

![](https://i.imgur.com/7N4FXaH.png)

となります。では、これをガウス関数でフィットしてみましょう。

```
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
```

mygaus という関数を定義し、フィットの初期値を探すために、測定データを再現できるようなモデルを表示してみます。

![](https://i.imgur.com/oQFBGVw.png)

```
chi2reg = Chi2Regression(mygauss,hist_x,hist_y)
fit = Minuit(chi2reg,mu=430,sigma=20,area=1000)
fit.migrad()
fit.minos() 

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
```

そこそこ良さそうな初期値を見つけたら、Chi-square で最適化してみます。

![](https://i.imgur.com/z3tMtrn.png)

みたいになります。

```
print(fit.print_param())
print(fit.values)
print(fit.errors)
```
とすると、フィットで得られたパラメータとその誤差を表示できます。

```
┌───┬───────┬───────────┬───────────┬────────────┬────────────┬─────────┬─────────┬───────┐
│   │ Name  │   Value   │ Hesse Err │ Minos Err- │ Minos Err+ │ Limit-  │ Limit+  │ Fixed │
├───┼───────┼───────────┼───────────┼────────────┼────────────┼─────────┼─────────┼───────┤
│ 0 │ mu    │  426.13   │   0.24    │   -0.24    │    0.24    │         │         │       │
│ 1 │ sigma │   20.76   │   0.23    │   -0.23    │    0.23    │         │         │       │
│ 2 │ area  │  2.073e3  │  0.021e3  │  -0.021e3  │  0.021e3   │         │         │       │
└───┴───────┴───────────┴───────────┴────────────┴────────────┴─────────┴─────────┴───────┘
None
<ValueView of Minuit at 7f92e8749060>
  mu: 426.134218801236
  sigma: 20.76215123588142
  area: 2073.1367177167863
<ErrorView of Minuit at 7f92e8749060>
  mu: 0.2432269180199041
  sigma: 0.23350886009026822
  area: 20.74372784277695
  ```

ガウス関数でフィットした中心値は 426.1+/-0.2 counts/sec ということで、最初に出した平均値 426.5 counts/s と $2\sigma$の範囲で一致しています。

また、ガウス関数の幅 20+/-0.2 counts/sec ですが、sqrt(426)~20.6 counts/sec に（当たり前ですが）一致します。もし、この測定が統計的なバラツキ以外に変動をもつなら、レート平均値の平方根よりも幅が広がります.
今回の測定では、統計的な変動だけであることがわかります。

### 時間差の分布

時系列データはいろいろな解析ができます。連続して観測されるイベント間の到来時刻差(delta-t distribution)は、`np.diff(df_137cs['sec'])` で得ることができます。

これを片対数プロットに描くと、次のように直線になります。つまり、指数関数になります。137Cs の放射線の崩壊はランダムで、到来時間差は指数関数になります。

![](https://i.imgur.com/cSRSh7p.png)

これは次のコードで描画し、指数関数でフィットできます。

```
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

```

さて、この指数関数のパラメータは、
```
┌───┬──────┬───────────┬───────────┬────────────┬────────────┬─────────┬─────────┬───────┐
│   │ Name │   Value   │ Hesse Err │ Minos Err- │ Minos Err+ │ Limit-  │ Limit+  │ Fixed │
├───┼──────┼───────────┼───────────┼────────────┼────────────┼─────────┼─────────┼───────┤
│ 0 │ mu   │2.350325e-3│0.000034e-3│-0.000034e-3│0.000034e-3 │         │         │       │
│ 1 │ norm │164.6167e3 │ 0.0021e3  │ -0.0021e3  │  0.0021e3  │         │         │       │
└───┴──────┴───────────┴───────────┴────────────┴────────────┴─────────┴─────────┴───────┘
None
<ValueView of Minuit at 7fbab4b31780>
  mu: 0.0023503245242608357
  norm: 164616.71575401127
<ErrorView of Minuit at 7fbab4b31780>
  mu: 3.3962473970311133e-08
  norm: 2.1123118964314895
(py3.8.6env1) [enoto@chromy:l
```
ということで、$\exp(-t/\mu)$ の $\mu=0.00235$ ですが、$1/\mu=425.5$ counts/sec とレートに対応することも確認できます。この delta-t distibution は本来ランダムな信号を検出器が正しく測定できているかなどの評価に使えます。

ちなみに、data frame を読み込むときの `sec` のフォーマットを間違うと、この delta-t ditribution は崩れることに注意してください。

## スペクトルの解析

次に放射線のスペクトルを描いてみます。今回は、統計が少し悪く誤差が見えるデータの方が解説しやすいので、50 秒のデータでまずは描いてみましょう。

```
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
```

![](https://i.imgur.com/JXCFzHu.png)

青が 137Cs を照射した場合で、オレンジ色が環境バックグラウンドの測定です。ADC channel で 60 以上の箇所は、バックグラウンドと137Cs は一致しています。

測定時間で割り算して（縦軸を counts から counts/sec　にする)、バックグラウンドを引いたデータについて、662 keV の輝線をガウス関数でフィットしてみます。

```
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
```
![](https://i.imgur.com/1woDu96.png)

今回のモデルでは、ガウス関数に加えて１次関数も加えています。図にはデータ点と、ベストフィットのモデル関数（赤）を示していて、下のパネルには残差を示しています。

最適のモデルパラメータは、
```
┌───┬───────┬───────────┬───────────┬────────────┬────────────┬─────────┬─────────┬───────┐
│   │ Name  │   Value   │ Hesse Err │ Minos Err- │ Minos Err+ │ Limit-  │ Limit+  │ Fixed │
├───┼───────┼───────────┼───────────┼────────────┼────────────┼─────────┼─────────┼───────┤
│ 0 │ mu    │   50.46   │   0.08    │   -0.07    │    0.07    │         │         │       │
│ 1 │ sigma │   1.93    │   0.09    │   -0.08    │    0.08    │         │         │       │
│ 2 │ area  │    99     │     4     │     -4     │     4      │         │         │       │
│ 3 │ c0    │    4.5    │    1.6    │    -1.6    │    1.6     │         │         │       │
│ 4 │ c1    │  -0.075   │   0.031   │   -0.030   │   0.030    │         │         │       │
└───┴───────┴───────────┴───────────┴────────────┴────────────┴─────────┴─────────┴───────┘
None
<ValueView of Minuit at 7fe516f08b20>
  mu: 50.463643473409775
  sigma: 1.9267523240788242
  area: 99.07492309277377
  c0: 4.527929611161785
  c1: -0.07519915368978516
<ErrorView of Minuit at 7fe516f08b20>
  mu: 0.07590302508801514
  sigma: 0.08543496081129959
  area: 4.3795997490523275
  c0: 1.6199947988577938
  c1: 0.03050768910717568
```
ということで、ガウス関数のピークは 50.464+/-0.076 で、幅は Sigma=1.927+/-0.085、このピークのレートは 99.1+/-4.4 cps とわかります。

### 137Cs の 662 keV の輝線の時間変動

最初のライトカーブはエネルギーなどで選別はしていませんでしたが、137Cs の 662 keV をガウス関数で近似した場合のピークと幅が求まったので、3$\sigma$ に入るイベントだけを抜き出した時間変動も描けるようになりました。

```
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
```

![](https://i.imgur.com/SZQYN3G.png)

このライトカーブでは、137Cs の 662 keV 輝線のレートから、同じエネルギー帯域（ADC channel)のバックグラウンドのレートの平均値を引いています。

- Average rate of the 662 keV line from 137 Cs = 125.6
- Average rate of the background = 18.1
- Subtracted average rate = 107.5

になります。ちなみに、スペクトルフィットでガウス関数と一次関数の和でモデル化したときのガウス関数のエリア（99 cps) と比べると 8 cps ほど多いのは、コンプトン等の成分が入っているから？？でしょうか。ここは未確認です。

### 60Co の 1173 keV と 1332 keV のライン

137Cs 同様に、60Co でのフィットを行うと以下のようになる。

![](https://i.imgur.com/SzsCoeV.png)

今回は、２つのガウス関数と連続成分は２次関数を仮定している。
```
┌───┬────────┬───────────┬───────────┬────────────┬────────────┬─────────┬─────────┬───────┐
│   │ Name   │   Value   │ Hesse Err │ Minos Err- │ Minos Err+ │ Limit-  │ Limit+  │ Fixed │
├───┼────────┼───────────┼───────────┼────────────┼────────────┼─────────┼─────────┼───────┤
│ 0 │ mu1    │   89.90   │   0.05    │   -0.05    │    0.05    │         │         │       │
│ 1 │ sigma1 │   2.46    │   0.07    │   -0.07    │    0.07    │         │         │       │
│ 2 │ area1  │    230    │    10     │     -9     │     10     │         │         │       │
│ 3 │ mu2    │  102.27   │   0.06    │   -0.06    │    0.06    │         │         │       │
│ 4 │ sigma2 │   2.48    │   0.07    │   -0.07    │    0.07    │         │         │       │
│ 5 │ area2  │    205    │     9     │     -9     │     9      │         │         │       │
│ 6 │ c0     │    -30    │    26     │    -26     │     27     │         │         │       │
│ 7 │ c1     │    1.1    │    0.6    │    -0.6    │    0.6     │         │         │       │
│ 8 │ c2     │  -0.0073  │  0.0030   │  -0.0029   │   0.0030   │         │         │       │
└───┴────────┴───────────┴───────────┴────────────┴────────────┴─────────┴─────────┴───────┘
None
<ValueView of Minuit at 7facab4500c0>
  mu1: 89.89885681279526
  sigma1: 2.4635201398282667
  area1: 230.37813109658913
  mu2: 102.27387615149468
  sigma2: 2.476543896033184
  area2: 205.07381204912983
  c0: -29.742310299289525
  c1: 1.095319696384393
  c2: -0.007302664336358126
<ErrorView of Minuit at 7facab4500c0>
  mu1: 0.046191615919659215
  sigma1: 0.07260118129707424
  area1: 9.756022732605876
  mu2: 0.055135061931318875
  sigma2: 0.07349858005025595
  area2: 8.936257391186867
  c0: 26.114650409287663
  c1: 0.565827150139758
  c2: 0.0029774089088468775
  ```

## ADC channel vs. Energy 

さて、３本のラインでフィットができたので、Energy (keV) と ADC channel の相関を見てみる。

| Ion | Energy (keV) | ADC channel | ADC error |
| ----- | -------| ------| -----|
| 137Cs | 661.65 | 50.46 | 0.08 |
| 60Co  | 1173.23| 89.90 | 0.05 |
| 60Co  | 1332.50| 102.27| 0.06 |

```
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
```

![](https://i.imgur.com/Ky6ySTf.png)

```
┌───┬──────┬───────────┬───────────┬────────────┬────────────┬─────────┬─────────┬───────┐
│   │ Name │   Value   │ Hesse Err │ Minos Err- │ Minos Err+ │ Limit-  │ Limit+  │ Fixed │
├───┼──────┼───────────┼───────────┼────────────┼────────────┼─────────┼─────────┼───────┤
│ 0 │ m    │ 77.20e-3  │  0.15e-3  │  -0.15e-3  │  0.15e-3   │         │         │       │
│ 1 │ c    │   -0.64   │   0.17    │   -0.17    │    0.17    │         │         │       │
└───┴──────┴───────────┴───────────┴────────────┴────────────┴─────────┴─────────┴───────┘
None
<ValueView of Minuit at 7fc4f78f78b0>
  m: 0.07720473893135693
  c: -0.6438362818461201
<ErrorView of Minuit at 7fc4f78f78b0>
  m: 0.00014678653797836546
  c: 0.1695069711635306
```

となるので、

$\frac{\textrm{ADC}}{\textrm{ch}} = 0.0772 \times \left(\frac{\textrm{Energy}}{\textrm{keV}}\right) - 0.644$

という関係にあることがわかる。2^10=1024 channel は約 13 MeV に対応することがわかる。

### 環境バックグラウンド

最後に、ADC channel をエネルギーに変換できるようになったので、X軸をエネルギー、Y軸を Counts/bin ではなく、Counts/sec/keV の次元にそろえてみる。

```
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
```

![](https://i.imgur.com/944u5jL.png)

などとなる。

## ROOT のようにヒストグラムに詰める

今回は、順を追って理解できるように、クラス化したり、メソッド(関数)化したりはしていません。使ってくると、同じ操作はまとめたくなります。

たとえば、上記のヒストグラムに詰める操作は、以下のようなクラスを定義すると、ROOT っぽい操作をすることもできます。

```python
class Hist1D(object):

  def __init__(self, nbins, xlow, xhigh):
    self.nbins = nbins
    self.xlow  = xlow
    self.xhigh = xhigh
    self.hist, edges = np.histogram([], bins=nbins, range=(xlow, xhigh))
    self.bins = (edges[:-1] + edges[1:]) / 2.

  def fill(self, arr):
    hist, edges = np.histogram(arr, bins=self.nbins, range=(self.xlow, self.xhigh))
    self.hist += hist

  @property
  def data(self):
    return self.bins, self.hist
```

を定義して、

```python
h = Hist1D(220,0,12.0)
h.fill(hdu_clevt['EVENTS'].data['PI']/100.)
plt.step(*h.data)
plt.xlim(0,3.0)
plt.xlabel('Energy (keV)');plt.ylabel('Count/bin')
plt.show()
```
なお、ここでの `du_clevt['EVENTS'].data['PI']` はイベントの array です。FITS 形式の説明のあとで使えるようになります。




 
## 参考文献

- [probfit](https://probfit.readthedocs.io/en/latest/index.html)
- [babar_python_tutorial/notebooks/04_Fitting.html](http://piti118.github.io/babar_python_tutorial/notebooks/04_Fitting.html) 
- [Kai-Feng Chen, "Introduction to Numerical Analysis"](https://hep1.phys.ntu.edu.tw/~kfjack/lecture/numerical/2018/305/lecture-305.pdf)
- http://www.sherrytowers.com/cowan_statistical_data_analysis.pdf
- [Python modeling and fitting packages](https://indico.cern.ch/event/834210/contributions/3539146/attachments/1906327/3148396/2019-09-11_Python_Modeling.pdf)


