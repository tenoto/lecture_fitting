# Python による放射線データの解析

[![hackmd-github-sync-badge](https://hackmd.io/H__Gt0oGQGub8l8cusKdww/badge)](https://hackmd.io/H__Gt0oGQGub8l8cusKdww)



 [tenoto/lecture_fitting@GitHub](https://github.com/tenoto/lecture_fitting)
 
## 測定セットアップ

- 放射線測定は、2020年11月9日、19日に、理化学研究所の榎戸研実験室(研究本館 610号室)で、茨城高専の村上悦基君、理研・榎戸研の沼澤正樹さんにやっていただきました。
- 放射線源として、ここでは 137Cs, 60Co の 2つの測定データと、放射線源を置かない場合の環境放射線バックグラウンドの測定データを扱う。これらの線源から放出される代表的なガンマ線輝線のエネルギーは[物理定数表](https://github.com/tenoto/repository/blob/master/docs/physical_const_entv190819j.pdf)にも記載している。
- 検出器は「雷雲プロジェクト」用に開発してきた、コガモ検出器(Compact Gamma-ray Monitor; CoGaMo)を用いた。5x5x15 cm の CsI(Tl) 結晶シンチレータと MPPC 光検出器を用いている。

![](https://i.imgur.com/W9NIwlX.jpg)

## 測定データと解析スクリプト

- 解析スクリプトは、[tenoto/lecture_fitting@GitHub](https://github.com/tenoto/lecture_fitting) からダウンロードできる。
- 測定データは(今回の実習用に加工したものを) Dropbox からダウンロードでき、GitHub からプルした lecture_fitting の直下に置いてください。



