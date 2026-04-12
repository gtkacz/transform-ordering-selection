# H5: Statistical Validation of the Pipeline Length Degradation Effect

## Placement suggestion

The first paragraph below (the result itself) is written for the **Results** section, after the existing paragraph about single-operation vs. multi-stage pipeline performance (currently the paragraph beginning "The relationship between preprocessing complexity..."). The second and third paragraphs are written for the **Discussion** section, after the paragraph about computational cost analysis.

## Required additions to refs.bib

```bibtex
@book{Good2005,
  author    = {Good, Phillip I.},
  title     = {Permutation, Parametric, and Bootstrap Tests of Hypotheses},
  edition   = {3rd},
  publisher = {Springer},
  address   = {New York},
  year      = {2005},
  doi       = {10.1007/b138696}
}
```

---

## LaTeX paragraphs — Results section

To determine whether the observed decline in mean $\alpha$ with increasing pipeline length reflects a genuine length-dependent mechanism or merely a combinatorial artifact arising from the larger number of possible orderings at greater lengths, a permutation test was conducted across the 64 non-baseline pipelines. The Pearson correlation between pipeline length (1 to 4 transformations) and $\alpha$ was $r = -0.421$, indicating a moderate negative association. Under the null hypothesis of no length effect, pipeline alpha values were randomly reassigned to length categories (preserving the group sizes of 4, 12, 24, and 24) across 100{,}000 permutations. Only 0.021\% of permuted datasets produced a correlation as extreme as the observed value ($p = 0.0002$), establishing that the negative association between pipeline length and accuracy gain is statistically significant and not attributable to the combinatorial structure of the experimental design. However, a separate test for strict monotonic decline across all four length levels yielded $p = 0.059$, indicating that while the overall negative trend is robust, the specific step-wise monotonicity from each length level to the next should be interpreted with appropriate caution. Consistent with this trend, the proportion of pipelines achieving positive $\alpha$ decreases monotonically with length: 75\% at length 1 (3 of 4), 42\% at length 2 (5 of 12), 21\% at length 3 (5 of 24), and 8\% at length 4 (2 of 24).

## LaTeX paragraphs — Discussion section

The permutation-validated pipeline length effect provides a quantitative foundation for the qualitative observation that simpler preprocessing strategies tend to outperform complex multi-stage alternatives. The combinatorial concern — that longer pipelines merely explore a larger space of possible orderings and are therefore more likely to include poor configurations by chance — is directly addressed by the permutation test, which holds group sizes constant while randomizing alpha assignments. The resulting $p = 0.0002$ demonstrates that the performance degradation with length cannot be explained by this combinatorial argument alone; rather, there exists an intrinsic cost to pipeline complexity that compounds with each additional transformation step. The most parsimonious explanation for this compounding effect is that each transformation in a multi-stage pipeline has a non-trivial probability of disrupting the feature representations established by preceding steps, and these disruption probabilities accumulate multiplicatively through the pipeline. Under this model, the probability that a length-$k$ pipeline preserves or improves feature quality decreases geometrically with $k$, consistent with the observed decline from 75\% beneficial pipelines at length 1 to 8\% at length 4.

This finding has immediate practical consequences for preprocessing pipeline design in medical imaging applications. The statistically validated length effect, combined with the cost-adjusted performance analysis through $\alpha_w$, establishes that single-transformation and two-transformation pipelines occupy a fundamentally different region of the performance-cost space than their three- and four-transformation counterparts. The data suggest that practitioners seeking to optimize preprocessing should first evaluate single transformations independently, select the best-performing individual operation, and only consider adding a second transformation if the marginal accuracy gain justifies the additional computational cost and the increased risk of feature-space interference. Extending pipelines beyond two transformations is unlikely to improve cost-adjusted performance and carries substantial risk of degradation, particularly when the added transformations include computationally expensive operations such as denoising. This evidence-based guideline stands in contrast to the common practice in the medical imaging literature of applying multi-stage preprocessing pipelines derived from domain intuition or convention without systematic evaluation of the marginal contribution of each stage.
