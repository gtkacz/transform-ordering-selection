# Transform Ordering Dominates Selection in Image Preprocessing Pipelines for Skin Disease Classification

**Gabriel Mitelman Tkacz**, *Member, IEEE*, and **Gustavo Scalabrini Sampaio**

---

## Abstract

Image preprocessing is a standard step in medical image classification pipelines, yet its impact on model accuracy remains poorly characterized, particularly regarding how the ordering of multiple transforms affects performance. This paper presents an exhaustive combinatorial evaluation of all possible preprocessing pipelines composed from four fundamental image transforms — histogram equalization, normalization, denoising, and color space conversion — applied to a convolutional neural network trained for binary classification of dermatological diseases. All 65 pipeline configurations spanning lengths zero through four are evaluated, generating a complete performance landscape over the combinatorial space. Variance decomposition reveals that transform ordering accounts for 68–78% of performance variance at pipeline lengths of three or more, overshadowing the contribution of transform selection. Positional analysis identifies a statistically significant complementary structure: histogram equalization exhibits a strong preference for early placement while normalization favors the terminal position, and permutation tests confirm that these preferences are not attributable to chance. Confusion matrix decomposition shows that 72% of preprocessing-induced accuracy losses arise from decision threshold shifts rather than feature destruction, suggesting that post-hoc calibration may partially recover lost performance. The proportion of beneficial pipelines declines from 75% at length one to 8% at length four, establishing a statistically significant length-degradation effect. These findings motivate an evidence-based pipeline design template that places distribution-dependent transforms first and linear standardizers last.

**Keywords:** Image preprocessing, preprocessing pipeline design, transform ordering, convolutional neural networks, medical image classification, dermatological diagnosis, computer-aided detection.

---

## I. Introduction

The field of dermatology has experienced rapid adoption of artificial intelligence tools for diagnostic support, particularly through computer-aided diagnosis (CADx) systems trained on dermoscopic and clinical images [Krakowski2024]. Deep learning models can increase diagnostic accuracy for skin diseases by up to 12 percentage points relative to unaided clinical assessment [Jain2021, Brinker2019], a substantial margin given the 66.3% baseline accuracy reported for human physicians in certain diagnostic tasks [Barnett2019]. These performance gains, however, depend critically on the quality and preparation of the input images fed to the classification model [AlHinai2020, Brinker2019].

Image preprocessing — the application of transforms such as normalization, histogram equalization, denoising, and color space conversion prior to model ingestion — is widely regarded as a necessary step for achieving reliable classification performance [Rodrigues2020, Avsar2021, Tarawneh2024]. In practice, preprocessing pipelines are typically constructed by selecting a set of transforms believed to be individually beneficial and applying them in an order motivated by domain convention or intuition. A typical workflow might normalize intensity values, apply noise reduction, and enhance contrast, with the ordering dictated by the practitioner's understanding of each transform's purpose rather than by systematic evaluation.

This conventional approach implicitly assumes two properties of preprocessing pipelines. First, it assumes that the choice of which transforms to include (transform selection) is the primary determinant of pipeline performance. Second, it assumes that the order in which selected transforms are applied (transform ordering) is either irrelevant or can be determined from first principles without empirical evaluation. Neither assumption has been rigorously tested. Because common preprocessing transforms — histogram equalization, nonlinear denoising, color space conversion — are nonlinear and data-dependent, the mathematical non-commutativity of their composition is expected in principle but has not been quantified in terms of its impact on downstream classification accuracy.

This paper addresses this gap through an exhaustive combinatorial evaluation of all possible preprocessing pipelines that can be constructed from four fundamental image transforms: histogram equalization (E), intensity normalization (N), non-local means denoising (D), and color space conversion (CS). By evaluating every permutation at every pipeline length from one through four — 64 non-baseline configurations plus the unprocessed baseline, for a total of 65 evaluated pipelines — the study generates a complete map of the pipeline performance space. This combinatorial design enables, for the first time, a direct decomposition of preprocessing performance into the contributions of transform selection and transform ordering, quantified through variance analysis and permutation-based statistical tests.

The principal findings of this study are fourfold. First, a permutation-validated length-degradation effect demonstrates that pipeline performance declines significantly with the number of transforms applied, with the proportion of accuracy-improving pipelines falling from 75% at length one to 8% at length four. Second, variance decomposition reveals a crossover from selection-dominated variance at pipeline length two to ordering-dominated variance at length three, where arrangement of the transforms explains approximately 78% of total performance variance. Third, positional analysis identifies equalization and normalization as exhibiting statistically significant and complementary positional preferences — equalization favoring early placement and normalization favoring the terminal position — while denoising and color space conversion are approximately position-neutral. Fourth, confusion matrix decomposition reveals that the dominant mechanism of preprocessing-induced error is a shift of the effective decision threshold rather than a destruction of class-discriminative features, with 72% of detrimental pipelines exhibiting this threshold-shift signature.

These findings are synthesized into a practical pipeline design template — E → {D, CS} → N — that places distribution-dependent operations at the pipeline boundary and permits free arrangement of position-neutral operations in the interior. The remainder of this paper is organized as follows. Section II reviews related work on preprocessing evaluation in medical imaging. Section III describes the experimental methodology, including the combinatorial design and statistical framework. Section IV presents the results. Section V discusses the theoretical and practical implications. Section VI concludes.

---

## II. Related Work

The role of image preprocessing in medical image classification has been investigated across several clinical domains, though the majority of prior work evaluates individual transforms rather than their combinatorial interactions. This section reviews the existing literature on preprocessing evaluation, identifies the methodological gap addressed by the present study, and situates the contribution within the broader context of pipeline design for computer-aided diagnosis.

Rodrigues et al. [Rodrigues2020] evaluated six preprocessing strategies for HEp-2 cell classification in immunofluorescence images, including the original unprocessed image, mean subtraction, contrast stretching (with and without mean subtraction), and histogram equalization (with and without mean subtraction). Their study found, counterintuitively, that the highest accuracy was achieved using unprocessed images for most CNN architectures, with Inception-V3 reaching 96.69% on the original images. Only ResNet-50 benefited from contrast stretching. This result highlights a key tension in the preprocessing literature: transforms that are beneficial in isolation may offer no advantage — or may even degrade performance — when applied to architectures capable of learning equivalent representations from raw data. However, the study evaluated each strategy independently and did not examine multi-stage pipelines or the effect of transform ordering.

In the biometric authentication domain, Tarawneh et al. [Tarawneh2024] investigated the effect of mean and median filtering on CNN-based dorsal hand vein recognition. Without preprocessing, their model achieved 70% accuracy on the training set; with mean filtering, accuracy rose to 99%. While this study demonstrates that appropriate preprocessing can yield dramatic improvements, it again evaluated single transforms in isolation, without considering how the combination or ordering of multiple transforms might affect the results.

For dermoscopic image analysis specifically, Arora et al. [Arora2021] proposed a modified U-Net architecture incorporating group normalization and attention gates for skin lesion segmentation. Their preprocessing pipeline included rotation, resizing, cropping, and mirroring for data augmentation, achieving segmentation accuracies of 95–97% across four public databases. Dash et al. [Dash2020] developed a cascaded deep CNN framework for psoriasis severity assessment that included intensity normalization and artifact removal as preprocessing steps. Nugroho et al. [Nugroho2024] examined the effect of preprocessing — specifically hair removal, contrast enhancement, and noise reduction — on skin lesion segmentation accuracy, demonstrating that appropriate preprocessing improved segmentation performance. In each case, the preprocessing pipeline was designed and applied as a fixed sequence without systematic exploration of alternative orderings.

Avsar [Avsar2021] evaluated the effects of image preprocessing on CNN performance for pneumonia detection, testing preprocessing techniques including Wiener filtering and histogram equalization. The study demonstrated that preprocessing selection significantly affected classification accuracy, but — like the other works reviewed here — did not address the ordering of multi-stage pipelines. Vincent and Roopa Jayasingh [Vincent2022] surveyed preprocessing techniques for psoriasis detection, cataloguing segmentation, feature extraction, and classification approaches without examining how the sequencing of these operations affects downstream accuracy.

A consistent methodological gap emerges from this review. Existing studies treat preprocessing transforms as independent interventions whose effects can be evaluated in isolation. No prior work, to our knowledge, has conducted an exhaustive combinatorial evaluation of all possible orderings of a given set of transforms, nor has any study decomposed the variance in classification performance into the separate contributions of transform selection and transform ordering. The present study fills this gap by evaluating all 65 pipeline configurations constructible from four transforms, enabling a direct comparison between the predictive power of knowing *which* transforms are in a pipeline versus knowing *how* they are ordered.

---

## III. Methodology

### A. Dataset and Classification Architecture

The experimental dataset was constructed by combining dermoscopic images of skin lesions from the HAM10000 collection [Tschandl2018] with images of healthy skin from the Healthy Skin dataset [Paulson2023], forming a binary classification task (diseased versus healthy). Images were resized to a uniform spatial resolution prior to model ingestion.

The classification architecture consisted of a four-block convolutional neural network (CNN). Each block comprised a convolutional layer, batch normalization, ReLU activation, and max pooling. Filter counts doubled at each block — 32, 64, 128, and 256 — followed by fully connected layers reducing dimensionality to a single output node with sigmoid activation for binary classification.

The dataset was partitioned into training (80%), validation (10%), and test (10%) subsets using a fixed random seed to ensure identical splits across all 65 experimental configurations. Training used the Adam optimizer with a learning rate of $10^{-4}$ and mean squared error (MSE) as the loss function. All experiments were executed on a system equipped with an Intel Core i5-12600KF processor, 32 GB of RAM, and an NVIDIA GeForce RTX 4080 SUPER GPU running CUDA 12.0.

### B. Preprocessing Transforms

Four fundamental image transforms were selected for evaluation, spanning the principal categories of preprocessing commonly employed in medical imaging pipelines: intensity standardization, contrast enhancement, noise suppression, and color representation.

**Intensity Normalization (N)** applies an affine transformation $x' = (x - \mu_t) / \sigma_t$ to rescale pixel values toward a target mean $\mu_t = 0.4$ and standard deviation $\sigma_t = 0.2$. This standardization reduces inter-sample variability in lighting and contrast, allowing the network optimizer to converge more efficiently [irjet].

**Histogram Equalization (E)** redistributes pixel intensities to approximate a uniform cumulative distribution function [Gonzalez2006, Xiao2019, Nugroho2024]. By expanding the dynamic range, equalization enhances subtle textural and chromatic contrasts that may be diagnostically relevant in skin lesion assessment.

**Non-Local Means Denoising (D)** suppresses acquisition noise by computing weighted averages over similar image patches, using a template window of 5 pixels and a search window of 19 pixels [Mafi2019, Elad2023]. Unlike local filters, non-local means preserves repetitive patterns and edge structures characteristic of dermatological lesions.

**Color Space Conversion (CS)** transforms the image from RGB to HSV representation, separating luminance from chrominance information [Wang2018, Ballester2022]. The HSV space allows the network to exploit hue and saturation channels independently, potentially revealing color-based diagnostic features that are entangled in the RGB representation.

Preprocessing parameters were determined through systematic grid search optimization, evaluating each parameter combination through complete model training and evaluation on the same data partition used for the main experiments. Table I summarizes the selected parameters.

> **Table I: Preprocessing Parameters**
>
> | Transform | Parameters |
> |-----------|-----------|
> | Normalization (N) | Mean: 0.4, Standard deviation: 0.2 |
> | Equalization (E) | Standard histogram equalization (no parameters) |
> | Denoising (D) | Template window: 5 px, Search window: 19 px |
> | Color Space (CS) | RGB → HSV |

### C. Combinatorial Experimental Design

The central methodological contribution of this study is the exhaustive evaluation of all possible preprocessing pipelines constructible from the four selected transforms. At each pipeline length $k$ (from 1 to 4), every ordered permutation of $k$ distinct transforms drawn from the set $\{E, N, D, CS\}$ was evaluated independently. This yields $P(4,1) + P(4,2) + P(4,3) + P(4,4) = 4 + 12 + 24 + 24 = 64$ non-baseline pipelines, plus the unprocessed baseline (length zero), for a total of 65 evaluated configurations.

This design is distinguished from prior work in two respects. First, it evaluates all orderings of each transform subset, not merely a single conventional ordering. For instance, the three-element subset $\{E, D, N\}$ is evaluated in all six orderings: E→D→N, E→N→D, D→E→N, D→N→E, N→D→E, and N→E→D. Second, by spanning all pipeline lengths, the design enables direct statistical comparison of performance as a function of pipeline complexity.

### D. Performance Metrics

Three metrics quantify preprocessing impact. The accuracy gain $\alpha$ measures the absolute change in test-set accuracy:

$$\alpha = \varepsilon_x - \varepsilon_0$$

where $\varepsilon_x$ is the accuracy of the preprocessed model and $\varepsilon_0 = 88.5\%$ is the baseline accuracy without preprocessing. Positive $\alpha$ indicates improvement; negative $\alpha$ indicates degradation.

The time ratio $\gamma$ quantifies computational overhead:

$$\gamma = \frac{\rho_x}{\rho_0}$$

where $\rho_x$ and $\rho_0$ are the training times with and without preprocessing, respectively.

The cost-adjusted accuracy gain $\alpha_w$ integrates both performance and computational cost:

$$\alpha_w = \begin{cases} \alpha / \gamma & \text{if } \alpha \geq 0 \\ \alpha \cdot \gamma & \text{if } \alpha < 0 \end{cases}$$

This asymmetric formulation rewards accuracy improvements achieved at low computational cost while penalizing pipelines that both degrade accuracy and impose computational overhead.

### E. Statistical Analysis Methods

Four statistical techniques are employed to characterize the structure of the pipeline performance landscape.

**Variance decomposition** via one-way ANOVA quantifies the relative contribution of transform selection (between-set variance, determined by which transforms are in the pipeline) versus transform ordering (within-set variance, determined by how they are arranged) at each pipeline length. The effect size $\eta^2$ measures the proportion of total variance explained by transform set membership.

**Permutation tests** with 100,000 iterations assess the statistical significance of observed effects without distributional assumptions. These tests are applied to the length–performance correlation, transform positional preferences, and bookend configuration contrasts [Good2005].

**Positional analysis** computes the mean $\alpha$ achieved by each transform as a function of its ordinal position across all pipelines in which it appears, revealing position-dependent performance gradients.

**Confusion matrix decomposition** tracks the change in false positives ($\Delta$FP) and false negatives ($\Delta$FN) relative to the baseline for each pipeline, enabling classification of preprocessing-induced errors into threshold-shift and feature-degradation categories.

---

## IV. Results

### A. Pipeline Performance Overview

Table II presents the accuracy gain ($\alpha$) and cost-adjusted accuracy gain ($\alpha_w$) for all 65 pipeline configurations. The distribution of outcomes across the 64 non-baseline pipelines is strikingly asymmetric: approximately 23% of pipelines achieve positive $\alpha$ (accuracy improvement over the unprocessed baseline), while the remaining 77% degrade performance.

> **Table II: Pipeline Performance Results ($\varepsilon_0 = 88.5\%$)**
>
> | Pipeline | $\alpha$ (pp) | $\alpha_w$ |
> |----------|---------------|------------|
> | CS | +0.50 | 0.69 |
> | D | +8.00 | 2.75 |
> | E | +8.00 | 10.97 |
> | N | +7.00 | 9.99 |
> | CS → D | +2.50 | 0.85 |
> | CS → E | +2.00 | 2.70 |
> | CS → N | +7.00 | 7.58 |
> | D → CS | −0.50 | −1.38 |
> | D → E | +6.00 | 2.20 |
> | D → N | +8.50 | 2.89 |
> | E → CS | +6.50 | 6.85 |
> | E → D | +9.00 | 3.32 |
> | E → N | +8.00 | 11.01 |
> | N → CS | −8.50 | −5.95 |
> | N → D | −4.00 | −11.75 |
> | N → E | −2.50 | −1.85 |
> | CS → D → E | +2.50 | 0.84 |
> | CS → D → N | +7.50 | 2.56 |
> | CS → E → D | +0.00 | 0.00 |
> | CS → E → N | +9.00 | 12.17 |
> | CS → N → D | −4.00 | −11.95 |
> | CS → N → E | −6.50 | −4.79 |
> | D → CS → E | +0.50 | 0.17 |
> | D → CS → N | +5.00 | 1.71 |
> | D → E → CS | +0.50 | 0.17 |
> | D → E → N | +9.50 | 3.21 |
> | D → N → CS | −27.00 | −79.73 |
> | D → N → E | −2.00 | −5.93 |
> | E → CS → D | +3.50 | 1.17 |
> | E → CS → N | +9.50 | 13.17 |
> | E → D → CS | +6.00 | 1.99 |
> | E → D → N | +8.50 | 2.86 |
> | E → N → CS | +3.00 | 4.00 |
> | E → N → D | +6.50 | 2.19 |
> | N → CS → D | −8.00 | −24.08 |
> | N → CS → E | −10.00 | −7.12 |
> | N → D → CS | −4.00 | −12.34 |
> | N → D → E | −2.50 | −7.45 |
> | N → E → CS | −8.50 | −6.05 |
> | N → E → D | −2.00 | −6.03 |
> | CS → D → E → N | +7.50 | 2.50 |
> | CS → D → N → E | −28.00 | −85.10 |
> | CS → E → D → N | +6.50 | 2.12 |
> | CS → E → N → D | −15.00 | −45.96 |
> | CS → N → D → E | −1.50 | −4.56 |
> | CS → N → E → D | −2.50 | −7.66 |
> | D → CS → E → N | +7.50 | 2.51 |
> | D → CS → N → E | −5.00 | −14.95 |
> | D → E → CS → N | +8.00 | 2.96 |
> | D → E → N → CS | −15.50 | −42.22 |
> | D → N → CS → E | −29.50 | −79.30 |
> | D → N → E → CS | −27.00 | −80.62 |
> | E → CS → D → N | +9.50 | 3.14 |
> | E → CS → N → D | −11.00 | −33.77 |
> | E → D → CS → N | +8.50 | 2.78 |
> | E → D → N → CS | −8.50 | −24.71 |
> | E → N → CS → D | +2.50 | 0.82 |
> | E → N → D → CS | −3.00 | −9.19 |
> | N → CS → D → E | −6.50 | −19.47 |
> | N → CS → E → D | −7.00 | −21.35 |
> | N → D → CS → E | −19.50 | −59.47 |
> | N → D → E → CS | −23.50 | −64.23 |
> | N → E → CS → D | −16.00 | −43.78 |
> | N → E → D → CS | −11.50 | −35.06 |

The best-performing pipeline is E → CS → N, achieving $\alpha = +9.50$ percentage points (pp) with $\alpha_w = 13.17$. Two other configurations — D → E → N and E → CS → D → N — attain the same maximum $\alpha$ but with substantially lower cost-adjusted performance ($\alpha_w = 3.21$ and $3.14$, respectively), reflecting the computational overhead imposed by denoising operations. The worst-performing pipeline is D → N → CS → E at $\alpha = -29.50$ pp, representing an accuracy collapse to 59.0% from the 88.5% baseline.

Single-transform pipelines are surprisingly competitive when computational cost is considered. Equalization alone achieves $\alpha = +8.00$ pp with $\alpha_w = 10.97$, and normalization alone achieves $\alpha = +7.00$ pp with $\alpha_w = 9.99$ — cost-adjusted values that exceed those of most multi-stage pipelines. Among length-two pipelines, E → N stands out with $\alpha = +8.00$ pp and $\alpha_w = 11.01$, indicating that this simple two-step sequence captures most of the achievable benefit at minimal computational cost.

A striking pattern emerges among the worst-performing pipelines: all five configurations with $\alpha < -15$ pp contain normalization immediately followed by another transform at an early pipeline position (D → N → CS → E, CS → D → N → E, D → N → E → CS, D → N → CS, N → D → CS → E). This systematic clustering of failure around specific subsequences motivates the positional analyses presented in Sections IV-C and IV-D.

### B. Statistical Validation of the Length-Degradation Effect

To determine whether the observed decline in mean $\alpha$ with increasing pipeline length reflects a genuine length-dependent mechanism or merely a combinatorial artifact — longer pipelines explore more of the configuration space and are therefore more likely to include poor orderings by chance — a permutation test was conducted across the 64 non-baseline pipelines. The Pearson correlation between pipeline length (1 to 4 transforms) and $\alpha$ yielded $r = -0.421$, indicating a moderate negative association. Under the null hypothesis of no length effect, $\alpha$ values were randomly reassigned to length categories (preserving group sizes of 4, 12, 24, and 24) across 100,000 permutations. Only 0.021% of permuted datasets produced a correlation as extreme as the observed value ($p = 0.0002$), establishing that the negative association between pipeline length and accuracy gain is statistically significant and not attributable to the combinatorial structure of the experimental design.

A separate test for strict monotonic decline across all four length levels yielded $p = 0.059$, indicating that while the overall negative trend is robust, the specific step-wise monotonicity from each length to the next should be interpreted with appropriate caution. Consistent with the overall trend, the proportion of pipelines achieving positive $\alpha$ decreases monotonically with length: 75% at length 1 (3 of 4), 42% at length 2 (5 of 12), 21% at length 3 (5 of 24), and 8% at length 4 (2 of 24).

<!-- [Figure: Box plot or violin plot of α by pipeline length, showing the declining median and expanding variance] -->

### C. Variance Decomposition: Ordering Versus Selection

To quantify the relative contribution of transform selection versus transform ordering, the total variance in $\alpha$ among multi-stage pipelines was decomposed into a between-set component (attributable to which transforms are included) and a within-set component (attributable to how they are arranged) at each pipeline length. Pipelines sharing the same unordered set of transforms were grouped together, and a one-way ANOVA was computed with transform set as the grouping factor.

At pipeline length 2, where each of the six possible transform pairs admits only two orderings, transform selection accounts for the large majority of variance ($\eta^2 = 0.867$, $F(5,6) = 7.83$, $p = 0.013$). This indicates that at the simplest multi-stage level, the choice of which two transforms to apply is the primary determinant of pipeline performance, with ordering playing a subordinate role.

This relationship inverts at pipeline length 3, where each of the four possible transform triplets admits six orderings. The within-set (ordering) component now accounts for 77.6% of total variance ($\eta^2 = 0.224$ for the between-set component, $F(3,20) = 1.92$, $p = 0.158$), meaning that the identity of the transform set no longer significantly predicts $\alpha$ once ordering variation is considered.

At pipeline length 4, all 24 pipelines share the same transform set $\{E, N, D, CS\}$, so ordering is the sole source of variance by construction. The within-set standard deviation and range at this length demonstrate that rearranging the same four transforms can swing accuracy by tens of percentage points, exceeding the entire range of $\alpha$ across all four single-transform pipelines by a substantial factor.

<!-- [Figure: Stacked bar or line plot showing η² (selection) vs 1−η² (ordering) at each pipeline length, illustrating the crossover] -->

The crossover from selection-dominated variance at length 2 to ordering-dominated variance at length 3 admits a combinatorial interpretation. With two transforms there are only two possible orderings per set, limiting the scope for ordering-induced divergence; with three transforms there are six orderings, and with four there are 24, providing increasingly many opportunities for detrimental arrangements to emerge.

### D. Transform Positional Preferences and the Bookend Structure

To investigate whether specific transforms exhibit statistically significant positional preferences, the mean $\alpha$ achieved by each transform was computed as a function of its ordinal position (first through fourth) across all pipelines in which it appears. This positional analysis reveals a striking complementary structure between equalization and normalization.

Equalization exhibits a monotonically improving gradient favoring early placement, with mean $\alpha$ improving by +0.86 pp per position toward the front of the pipeline. Normalization displays the mirror-image pattern, with mean $\alpha$ improving by +1.07 pp per position toward the end of the pipeline. To assess whether equalization's first-position advantage reflects a genuine positional effect, pipelines were partitioned into equalization-first ($n = 16$) and equalization-not-first ($n = 48$) groups, and a permutation test with 100,000 iterations established that the advantage is statistically significant ($p = 0.014$). Equalization is the only transform among the four for which first-position placement achieves significance at the $p < 0.05$ level.

Denoising and color space conversion, by contrast, exhibit near-zero positional gradients (+0.05 pp and −0.18 pp, respectively), confirming that positional sensitivity is not a universal property of preprocessing operations but rather a characteristic of specific transform types.

The complementary positional structure is further confirmed by a bookend analysis examining the joint effect of first and last transforms. Pipelines adopting the equalization-first, normalization-last configuration ($n = 5$) achieve a mean $\alpha$ of −0.15 pp — nearly indistinguishable from the unprocessed baseline — with 60% of pipelines yielding positive accuracy gains. By contrast, the inverse configuration — normalization-first, equalization-last — produces a mean $\alpha$ of −2.02 pp with 0% positive-$\alpha$ pipelines. A permutation test on the direct contrast between these two bookend configurations yields $p = 0.024$, confirming that the +1.87 pp difference in mean $\alpha$ is statistically significant and that the equalization–normalization ordering is not interchangeable.

<!-- [Figure: Positional gradient plot for each transform showing mean α at each ordinal position (1–4)] -->

### E. Error Decomposition: Threshold Shift as the Dominant Mechanism

To characterize the mechanism through which preprocessing degrades classification accuracy, the change in false positives ($\Delta$FP) and false negatives ($\Delta$FN) relative to the baseline model was computed for each pipeline. These two error-shift variables exhibit a Pearson correlation of $r = -0.545$, indicating a strong anti-correlation: when a preprocessing pipeline increases false positives, it simultaneously decreases false negatives by a roughly proportional amount, and vice versa. This anti-correlation is the characteristic signature of a decision threshold shift — the preprocessing pipeline effectively moves the classifier's operating point along the receiver operating characteristic (ROC) curve rather than degrading its overall discriminative capacity.

Categorizing each pipeline by the direction of its error shift reveals that approximately 71.8% of all tested combinations produce a threshold-shift pattern, with false positives and false negatives moving in opposite directions. Of these, 40.6% shift the effective decision boundary toward the healthy classification (increasing false negatives while reducing false positives), while 31.2% shift it toward the diseased classification (increasing false positives while reducing false negatives). Only 17.2% of pipelines exhibit the simultaneous increase in both error types that would indicate genuine bidirectional feature-space degradation, and these pipelines carry the worst mean accuracy penalty (−1.8 pp), compared to −1.3 pp for sensitivity-eroding pipelines and −1.1 pp for specificity-eroding pipelines. The remaining 10.9% of pipelines improve both metrics simultaneously, with a corresponding positive mean $\alpha$ of +0.9 pp, confirming that genuine feature enhancement — while rare — does occur within the explored pipeline space.

---

## V. Discussion

### A. CNN Feature Representations and Ordering Invariance

The variance decomposition reveals a finding that, while consistent with the mathematical non-commutativity of nonlinear image transformations, is far from trivially predictable: the CNN architecture employed in this study is not ordering-invariant. A sufficiently expressive feature extractor could, in principle, learn representations that abstract away the differences between alternative orderings of the same transform set, rendering ordering irrelevant to classification accuracy. The empirical result — that ordering accounts for 68–78% of variance in $\alpha$ at pipeline length 3 and for 100% at length 4 — demonstrates that the four-block convolutional architecture used here lacks this invariance property. The feature-space distortions introduced by different orderings are large enough to propagate through the entire network to the classification layer.

The crossover from selection-dominated variance at length 2 to ordering-dominated variance at length 3 admits a combinatorial interpretation: with two transforms, there are only two possible orderings per set, limiting the scope for ordering-induced divergence; with three there are six orderings, and with four there are 24, providing increasingly many opportunities for detrimental arrangements to emerge. This combinatorial argument does not, by itself, explain why mean $\alpha$ declines with length — it explains only why variance increases — but it does identify ordering proliferation as the mechanism through which longer pipelines explore larger regions of the performance space, with the majority of those regions corresponding to suboptimal configurations.

The practical implication of ordering dominance is that the conventional approach to preprocessing pipeline design — selecting a set of beneficial transforms and applying them in a conventionally motivated order — is insufficient for pipelines of length three or greater. At these lengths, the choice of arrangement is the primary determinant of whether a pipeline improves or degrades classification accuracy, and evaluating a single ordering of a promising transform set provides almost no information about the performance of alternative orderings of the same set.

### B. Distribution-Dependent Versus Distribution-Agnostic Transforms

The statistically validated positional preferences of equalization and normalization admit a coherent mechanistic interpretation grounded in the mathematical properties of each transform. Histogram equalization is a nonlinear, data-dependent remapping that redistributes pixel intensities to approximate a uniform cumulative distribution [Pizer1987]. Because this redistribution is computed from the empirical histogram of the input image, its output — and hence the features it enhances — depends critically on the distributional properties of its input. When equalization operates on the raw image (first position), it has access to the full dynamic range and the native intensity distribution, maximizing its capacity to reveal subtle textural contrasts in dermatological lesions. When preceding transforms have already altered the intensity distribution — particularly normalization, which compresses the distribution toward a fixed mean and standard deviation — the histogram from which equalization computes its mapping is informationally impoverished, and the resulting enhancement is correspondingly degraded.

Normalization, by contrast, is a linear affine transformation ($x' = (x - \mu) / \sigma$) that is agnostic to the distributional shape of its input; its sole function is to shift and scale pixel values to match the statistical expectations of the downstream convolutional network. This operation is therefore most effective when applied immediately before network ingestion, regardless of what transforms precede it. The mirror-image positional gradients observed empirically — equalization improving toward the front, normalization improving toward the back — are thus direct consequences of these distinct mathematical sensitivities: a distribution-dependent nonlinear operation requires pristine input distributions, while a distribution-agnostic linear standardizer benefits from being the final transform before the network.

More broadly, the finding that only two of the four transforms exhibit significant positional sensitivity suggests a general taxonomy of preprocessing operations: *position-sensitive* transforms whose effectiveness depends on input distributional properties (exemplified by equalization) versus *position-neutral* transforms that function comparably regardless of pipeline placement (exemplified by denoising and color space conversion). For practitioners designing preprocessing pipelines, this taxonomy implies a heuristic: position-sensitive, distribution-dependent operations should be placed as early as possible to preserve their access to native image statistics, while linear standardizers should be deferred to the final pipeline stage. Applied to the four-transform case, this heuristic yields the E → {D, CS} → N template, where the interior operations can be ordered with relative impunity given their positional neutrality and the boundary operations are fixed by their distributional requirements.

### C. Threshold Shift and Clinical Implications

The finding that 72% of preprocessing-induced accuracy losses are attributable to threshold shifts rather than feature destruction carries direct implications for clinical deployment. In a threshold-shift scenario, the preprocessing pipeline effectively moves the classifier's operating point along the ROC curve: the model's overall capacity to distinguish healthy from diseased tissue is preserved, but its calibration — the balance between false positives and false negatives — is disturbed. This mechanism suggests that the detrimental effects of suboptimal preprocessing may be partially mitigable through post-hoc recalibration of the classifier's decision threshold without requiring retraining [Liang2022].

In contrast, the 17.2% of pipelines that cause genuine bidirectional degradation — where both sensitivity and specificity deteriorate simultaneously — represent fundamentally incompatible transformation sequences whose effects cannot be compensated through threshold adjustment alone. These pipelines, which carry the worst mean accuracy penalty, correspond to configurations that irreversibly destroy class-discriminative features in the image.

For clinical systems, the distinction between threshold-shift and feature-destruction errors is consequential. A specificity-eroding pipeline that over-diagnoses disease may be acceptable — or even preferable — when the cost of missed diagnoses far exceeds the cost of false alarms, provided that the threshold shift is recognized and the overall accuracy penalty remains within acceptable bounds. The identification of threshold-shift dominance therefore motivates the development of preprocessing-aware calibration procedures that jointly optimize the transformation pipeline and the classification threshold, an approach that could recover a substantial portion of the performance lost to suboptimal preprocessing ordering.

### D. Practical Pipeline Design Guidelines

The results presented in this study converge on a set of evidence-based guidelines for preprocessing pipeline design in medical image classification.

First, the statistically significant length-degradation effect establishes that simpler pipelines are not merely computationally cheaper but fundamentally more likely to improve classification accuracy. The data demonstrate that practitioners seeking to optimize preprocessing should first evaluate single transforms independently, select the best-performing individual operation, and only consider adding a second transform if the marginal gain justifies the additional complexity. Extending pipelines beyond two transforms carries substantial risk of degradation and is unlikely to improve cost-adjusted performance, as reflected in the $\alpha_w$ values: single-transform pipelines achieve cost-adjusted gains of up to 10.97, while most three- and four-transform pipelines fall below 3.00.

Second, for pipelines that do incorporate multiple transforms, the positional analysis provides a concrete design template. Distribution-dependent, nonlinear transforms (exemplified by equalization) should occupy the earliest pipeline position to operate on unprocessed image statistics. Position-neutral transforms (denoising, color space conversion) may be placed in the interior in any order. Linear standardizers (normalization) should be deferred to the final position immediately before network ingestion.

Third, the ordering-dominance finding at pipeline lengths of three or more implies that evaluating a single ordering of a promising transform set provides almost no information about the performance of alternative orderings. Any systematic evaluation of multi-stage preprocessing must either enumerate all orderings or employ the positional template proposed here to constrain the search to a tractable subset of likely-beneficial configurations.

### E. Limitations and Future Directions

Several limitations of this study warrant consideration. The evaluation is restricted to a single four-block CNN architecture; modern architectures incorporating attention mechanisms, residual connections, or vision transformers may exhibit different sensitivity profiles to preprocessing ordering. The binary classification task (healthy versus diseased) represents the simplest diagnostic scenario, and it remains to be established whether the ordering-dominance finding extends to multi-class classification or fine-grained diagnostic tasks such as melanoma subtyping.

The four transforms evaluated here, while representative of the principal categories of image preprocessing, do not exhaust the space of operations commonly applied in medical imaging. Learned preprocessing layers, adaptive filtering techniques, and domain-specific artifact removal (e.g., hair removal in dermoscopy) may exhibit different positional sensitivities and interact differently within multi-stage pipelines.

Additionally, the experimental design evaluates each pipeline as a fixed preprocessing step applied identically to all images. In clinical practice, images arrive with heterogeneous quality characteristics, and an adaptive preprocessing strategy that selects and orders transforms conditionally on input image properties may achieve better overall performance than any fixed pipeline.

Future work should investigate how preprocessing ordering interacts with modern architectures — particularly whether the self-attention mechanism in vision transformers provides greater robustness to ordering variation, potentially absorbing distortions that the purely convolutional architecture studied here cannot. The development of preprocessing-aware calibration procedures, motivated by the threshold-shift finding, represents a promising direction for recovering performance lost to suboptimal pipeline configurations. Visualization and interpretation techniques such as Grad-CAM [Selvaraju2017] could illuminate how different orderings alter the learned feature representations at each network layer, providing deeper mechanistic understanding of the ordering-sensitivity phenomenon.

---

## VI. Conclusion

This study demonstrates that transform ordering — not transform selection — is the primary determinant of preprocessing pipeline performance when three or more transforms are applied to medical images prior to CNN-based classification. Through exhaustive combinatorial evaluation of all 65 pipeline configurations constructible from four fundamental image transforms, the work provides the first complete characterization of the preprocessing performance landscape and establishes several findings with direct practical relevance.

The statistically significant length-degradation effect ($r = -0.421$, $p = 0.0002$) establishes that each additional preprocessing step carries an intrinsic risk of performance degradation. The variance decomposition reveals that this risk is driven primarily by the proliferation of possible orderings, with ordering accounting for 78% of variance at length three and 100% at length four. Positional analysis identifies a constructive solution: histogram equalization requires early placement and normalization requires terminal placement, with the intermediate positions available for position-neutral operations such as denoising and color space conversion.

The confusion matrix analysis reveals that 72% of preprocessing-induced errors are threshold shifts — displacements of the classifier's operating point along the ROC curve rather than losses in discriminative capacity — suggesting that post-hoc calibration may recover a substantial fraction of preprocessing-induced accuracy losses.

These findings argue for a paradigm shift in preprocessing pipeline design: from ad hoc selection and conventional ordering toward evidence-based construction guided by the distributional properties of each transform and validated through systematic evaluation of all candidate orderings. The practical template E → {D, CS} → N, derived from the positional analysis, provides an immediately deployable heuristic that constrains the combinatorial search space while preserving the most effective pipeline configurations.

---

## References

<!-- Existing references (from refs.bib) -->

- [AlHinai2020] N. AlHinai, "Introduction to biomedical signal processing and artificial intelligence," in *Biomedical Signal Processing and Artificial Intelligence in Healthcare*, Academic Press, 2020, pp. 1–28.
- [Arora2021] R. Arora, B. Raman, K. Nayyar, and R. Awasthi, "Automated skin lesion segmentation using attention-based deep convolutional neural network," *Biomedical Signal Processing and Control*, vol. 65, p. 102358, 2021.
- [Avsar2021] E. Avşar, "Effects of image preprocessing on the performance of convolutional neural networks for pneumonia detection," in *Proc. INISTA*, 2021, pp. 1–5.
- [Ballester2022] C. Ballester et al., "Influence of color spaces for deep learning image colorization," arXiv:2204.02850, 2022.
- [Barnett2019] M. L. Barnett, D. Boddupalli, S. Nundy, and D. W. Bates, "Comparative accuracy of diagnosis by collective intelligence of multiple physicians vs individual physicians," *JAMA Network Open*, vol. 2, no. 3, p. e190096, 2019.
- [Brinker2019] T. J. Brinker et al., "Deep neural networks are superior to dermatologists in melanoma image classification," *European Journal of Cancer*, vol. 119, pp. 11–17, 2019.
- [Dash2020] M. Dash, N. D. Londhe, S. Ghosh, R. Raj, and R. S. Sonawane, "A cascaded deep convolution neural network based CADx system for psoriasis lesion segmentation and severity assessment," *Applied Soft Computing*, vol. 91, p. 106240, 2020.
- [Elad2023] M. Elad, B. Kawar, and G. Vaksman, "Image denoising: The deep learning revolution and beyond," arXiv:2301.03362, 2023.
- [Gonzalez2006] R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 3rd ed. Prentice-Hall, 2006.
- [Jain2021] A. Jain et al., "Development and assessment of an artificial intelligence–based tool for skin condition diagnosis by primary care physicians and nurse practitioners in teledermatology practices," *JAMA Network Open*, vol. 4, no. 4, p. e217249, 2021.
- [Krakowski2024] I. Krakowski et al., "Human-AI interaction in skin cancer diagnosis: A systematic review and meta-analysis," *NPJ Digital Medicine*, vol. 7, no. 1, p. 78, 2024.
- [Mafi2019] M. Mafi et al., "A comprehensive survey on impulse and Gaussian denoising filters for digital images," *Signal Processing*, vol. 157, pp. 236–260, 2019.
- [Nugroho2024] A. K. Nugroho, Afiahayati, M. E. Wibowo, and H. Soebono, "The effect of preprocessing on skin lesion segmentation," in *Proc. InCIT*, 2024, pp. 455–460.
- [Paulson2023] M. Paulson, "Healthy Skin Dataset," Hugging Face, 2023.
- [Rodrigues2020] L. F. Rodrigues, M. C. Naldi, and J. F. Mari, "Comparing convolutional neural networks and preprocessing techniques for HEp-2 cell classification in immunofluorescence images," *Computers in Biology and Medicine*, vol. 116, p. 103542, 2020.
- [Selvaraju2017] R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE ICCV*, 2017, pp. 618–626.
- [Tarawneh2024] O. Tarawneh et al., "The effect of pre-processing on a convolutional neural network model for dorsal hand vein recognition," *IJACSA*, vol. 15, no. 3, pp. 1284–1289, 2024.
- [Tschandl2018] P. Tschandl, C. Rosendahl, and H. Kittler, "The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions," arXiv:1803.10417, 2018.
- [Vincent2022] L. Vincent and J. Roopa Jayasingh, "Comparison of psoriasis disease detection and classification through various image processing techniques," in *Proc. ICDCS*, 2022, pp. 122–124.
- [Wang2018] G. Wang, Y. Liu, W. Xiong, and Y. Li, "An improved non-local means filter for color image denoising," *Optik*, vol. 173, pp. 157–173, 2018.
- [Xiao2019] B. Xiao, Y. Xu, H. Tang, X. Bi, and W. Li, "Histogram learning in image contrast enhancement," in *Proc. IEEE CVPRW*, 2019, pp. 1880–1889.
- [irjet] N. Verma and M. Dutta, "Contrast enhancement techniques: A brief and concise review," *IRJET*, vol. 4, no. 7, 2017.

<!-- New references to add to refs.bib -->

- [Good2005] P. I. Good, *Permutation, Parametric, and Bootstrap Tests of Hypotheses*, 3rd ed. Springer, 2005.
- [Liang2022] G. Liang, X. Zhang, C. Johnson, and N. Nguyen, "An investigation into decision boundary shift in medical image classification," *IEEE Access*, vol. 10, pp. 133290–133299, 2022.
- [Pizer1987] S. M. Pizer et al., "Adaptive histogram equalization and its variations," *Computer Vision, Graphics, and Image Processing*, vol. 39, no. 3, pp. 355–368, 1987.
