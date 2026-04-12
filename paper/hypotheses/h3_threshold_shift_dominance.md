# H3': Decision Threshold Shift as the Dominant Mechanism of Preprocessing-Induced Error

## Placement suggestion

These paragraphs are written for the **Discussion** section, immediately after the existing paragraph about "patterns of preprocessing failure" (currently the 4th paragraph of the Discussion). They provide a mechanistic decomposition of *how* detrimental pipelines degrade accuracy, building on the confusion matrix data already collected.

## Required additions to refs.bib

No new citations are strictly required — the analysis is self-contained from the existing experimental data. If desired, the following could support the threshold-shift framing:

```bibtex
@article{Liang2022,
  author  = {Liang, Gongbo and Zhang, Xin and Johnson, Connor and Nguyen, Nam},
  title   = {An Investigation Into Decision Boundary Shift in Medical Image Classification},
  journal = {IEEE Access},
  year    = {2022},
  volume  = {10},
  pages   = {133290--133299},
  doi     = {10.1109/ACCESS.2022.3231174}
}
```

---

## LaTeX paragraphs

Further insight into the mechanisms underlying preprocessing-induced performance degradation emerges from a systematic decomposition of the confusion matrix shifts observed across all 64 preprocessing pipelines. For each pipeline, the change in false positives ($\Delta\text{FP}$) and false negatives ($\Delta\text{FN}$) relative to the baseline model was computed. These two error-shift variables exhibit a Pearson correlation of $r = -0.545$, indicating a strong anti-correlation. This anti-correlation is the characteristic signature of a decision threshold shift rather than a generalized degradation of the model's learned feature representations: when a preprocessing pipeline increases the number of false positives, it simultaneously decreases false negatives by a roughly proportional amount, and vice versa. In a scenario where preprocessing genuinely destroys class-discriminative features, one would instead expect $\Delta\text{FP}$ and $\Delta\text{FN}$ to be uncorrelated or positively correlated, as the model would lose the ability to correctly classify samples from both categories.

Categorizing each pipeline by the direction of its error shift reveals that approximately 71.8\% of all tested combinations produce a threshold-shift pattern, with false positives and false negatives moving in opposite directions. Of these, 40.6\% shift the effective decision boundary toward the healthy classification (increasing false negatives while reducing false positives), while 31.2\% shift it toward the diseased classification (increasing false positives while reducing false negatives). Only 17.2\% of pipelines exhibit the simultaneous increase in both error types that would indicate genuine bidirectional feature-space degradation, and these pipelines carry the worst mean accuracy penalty ($\overline{\alpha} = -0.018$), compared to $\overline{\alpha} = -0.013$ for sensitivity-eroding pipelines and $\overline{\alpha} = -0.011$ for specificity-eroding pipelines. The remaining 10.9\% of pipelines improve both metrics simultaneously, with a corresponding positive $\overline{\alpha} = +0.009$, confirming that genuine feature enhancement — while rare — does occur within the explored pipeline space.

This finding carries direct implications for the clinical deployment of preprocessing pipelines. The observation that the majority of preprocessing-induced accuracy losses are attributable to threshold shifts rather than feature destruction suggests that the detrimental effects of suboptimal preprocessing may be partially mitigable through post-hoc calibration of the classifier's decision threshold, without requiring retraining. In contrast, the 17.2\% of pipelines that cause genuine bidirectional degradation — where both sensitivity and specificity deteriorate simultaneously — represent fundamentally incompatible transformation sequences whose effects cannot be compensated through threshold adjustment alone. For clinical systems, this distinction is consequential: a specificity-eroding pipeline that over-diagnoses disease may be acceptable (or even preferable) when the cost of missed diagnoses far exceeds the cost of false alarms, provided that the threshold shift is recognized and the overall accuracy penalty remains within acceptable bounds. The identification of threshold-shift dominance therefore motivates the development of preprocessing-aware calibration procedures that jointly optimize the transformation pipeline and the classification threshold, an approach that could recover a substantial portion of the performance lost to suboptimal preprocessing ordering.
