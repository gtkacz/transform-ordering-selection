<center>
    <p align="center">
        <img src="https://logodownload.org/wp-content/uploads/2017/09/mackenzie-logo-3.png" style="height: 7ch;"><br>
        <h1 align="center">Significance of Transform Ordering and Selection in CNN Preprocessing for Binary Skin Classification</h1>
        <h4 align="center">Gabriel Mitelman Tkacz, Gustavo Scalabrini Sampaio, Leandro Augusto da Silva</a></h4>
        <h4 align="center">Mackenzie Presbyterian University &mdash; S&atilde;o Paulo, Brazil</h4>
    </p>
</center>

<hr>

## Abstract

Image preprocessing is a near-universal step in medical image classification pipelines, yet its impact on model accuracy remains insufficiently characterized, particularly with respect to the ordering of multi-transform compositions. This study reports an exhaustive combinatorial evaluation of all 65 pipelines constructible from four fundamental transforms &mdash; histogram equalization, intensity normalization, non-local means denoising, and color space conversion &mdash; applied to a four-block convolutional neural network trained on the binary healthy-versus-diseased dermatoscopic classification task formed by using a dataset with 20,000 balanced images. Every pipeline was trained from scratch under five independent random seeds. Against a near-ceiling baseline of 98.09% test accuracy, which arithmetically bounds the detectable positive-&alpha; regime to at most +1.91 pp, no pipeline of length 2 or greater achieves a positive mean accuracy gain, and only equalization alone produces a seed-robust improvement. Holm-corrected permutation tests establish three ordering regularities: pipeline length correlates negatively with accuracy gain (*r* = &minus;0.44), equalization-first placement outperforms alternative first-position assignments by 1.10 pp (Cohen's *d* = 1.22), and the bookend configuration with equalization first and normalization last outperforms its mirror image by 2.00 pp (Cohen's *d* = 2.38). At pipeline *k* = 3, ordering accounts for 58% of accuracy-gain variance and transform selection for 42%. Confusion-matrix decomposition indicates that preprocessing-induced degradation is dominated by bidirectional feature-space damage (78%) rather than recoverable decision-threshold shift (22%). Within the scope of this single network, single binary task, single dataset, and single transform set, the evidence supports a cautious harm-minimization heuristic &mdash; equalization first, normalization last &mdash; rather than a general claim about ordering dominance in medical image preprocessing.

## Keywords

Image preprocessing &middot; Transform ordering &middot; Preprocessing pipeline design &middot; Convolutional neural networks &middot; Medical image classification &middot; Dermatological diagnosis
