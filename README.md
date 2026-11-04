# Predictive Modeling with Asymmetric Loss Functions

**A statistical learning study of pediatric heart-volume prediction using least-squares and quantile-regression models.**

**Author:** Panagiotis Anastasakis\
**Supervisor:** Dr. Yiannis Kamarianakis\
**Department:** Mathematics & Applied Mathematics, University of Crete\
**Thesis date:** June, 2023

------------------------------------------------------------------------

## 1. Project Overview

This project investigates how linear predictive models can account for situations in which different kinds of prediction errors have different consequences. Conventional least-squares regression treats errors symmetrically: an over-prediction and an under-prediction of equal magnitude receive the same penalty. In some applications, however, that assumption is not appropriate.

The motivating application is the prediction of **heart volume in pediatric patients that are candidates for heart transplantation**. Before transplantation, it can be difficult to measure a patient's heart volume accurately, yet an estimate can help clinicians assess whether a donor heart is appropriately sized.

The clinical motivation is asymmetric. An oversized graft may cause compression-related complications, while an undersized graft may provide insufficient cardiac output and is considered as the more dangerous direction of error. The study therefore examines **quantile regression**, which allows the fitting procedure to penalize under-prediction more heavily than over-prediction.

Note that this work is a statistical investigation and methodological comparison, not a deployable clinical decision system. Any model output would need to be considered alongside other clinical information and the judgement of medical professionals.

### Main questions

-   How do ordinary and penalized least-squares methods perform?
-   Can quantile regression better reflect the asymmetric cost of under-predicting heart volume?
-   How do alternative R implementations of penalized quantile regression compare?
-   What do repeated validation, permutation tests and bootstrap intervals reveal about model performance and uncertainty?

## 2. Data and Preparation

The application uses somatometric measurements and corresponding heart-volume observations. The original dataset contains **58 observations** and six candidate predictors:

  Variable   Description                  Unit / coding
  ---------- ---------------------------- ----------------------
  `Male`     Patient sex indicator        1 = male, 0 = female
  `Age`      Patient age                  months
  `Ht`       Height                       cm
  `Wt`       Weight                       kg
  `BMI`      Body mass index              kg/m²
  `BSA`      Body surface area            m²
  `HtVol`    **Response:** heart volume   mL

Four observations correspond to patients older than 18 years (216 months). They are excluded because the application focuses on pediatric patients, leaving **54 observations**.

### Exploratory analysis and feature construction

The exploratory analysis identifies increasing relationships between the continuous somatometric measurements and heart volume. Furthermore, a difference in average heart volume between female patients (471 ± 169 mL) and male patients (559 ± 277 mL) is reported in this sample, highlighting the relevance of gender as a heart volume predictor.

The predictors are strongly correlated. In particular, the reported Pearson and Spearman correlations show high positive associations among age, height, weight, BMI and BSA. Variance Inflation Factors (VIFs) are used to assess and mitigate multicollinearity. BSA is removed because first, it has an especially large VIF and second, because its estimation can vary across formulas making it an unreliable factor to include in the model. Collinearity remains among the remaining predictors, but instead of opting for a more aggressive filtering, a richer candidate feature set is constructed consisting of interactions and second-order terms of the initial features, where stepwise VIF filtering is then applied.

The new predictor set includes:

-   The initial variables, excluding BSA.
-   Pairwise interactions.
-   Second-order terms for continuous predictors.
-   Selected ratios of height and weight to age.

New features are created before standardization. Continuous predictors are standardized using z-scores; the categorical `Male` indicator is not standardized. Interactions involving `Male` are formed so that female observations retain zero in those interaction columns.

The stepwise VIF procedure repeatedly removes the predictor with the largest VIF until all remaining VIFs are at or below 10. The resulting modelling feature set is:

`Male`, `Age²`, `Ht²`, `Wt²`, `Ht/Age`, `Wt/Age`, `Male × Age` and `Male × BMI`.

This feature engineering is intended to relax simple linear-model assumptions while controlling severe multicollinearity.

## 3. Modeling Strategy

The Thesis compares two broad families of linear models.

### Least-squares models

The symmetric-loss baseline includes:

-   **Linear regression** as the conventional least-squares model.
-   **Subset selection**, specifically stepwise (forward & backward) approaches.
-   **Model averaging** based on the corrected Akaike Information Criterion (AICc), to reduce dependence on selecting a single model.
-   **Ridge regression**, which shrinks coefficients through an L2 penalty.
-   **Lasso regression**, which uses an L1 penalty and can shrink coefficients to zero.

The study also considers a logarithmic transformation of the response for two reasons. First, it allows the Least Squares models to focus on relative rather than absolute prediction errors, which is useful when heart volumes vary considerably in magnitude. Second, a concave transformation serves as a mitigator for any issues caused by the potential presence of hereroscedasticity.

### Quantile regression

Quantile regression estimates a conditional quantile rather than the conditional mean. It uses the pinball loss:

$$\rho_\tau(u) = u \bigl(\tau - \mathbb{1}(u < 0)\bigr),$$

where $u$ is the prediction error and $\tau$ is the target quantile.

At $\tau=0.5$, the method corresponds to median regression. At $\tau=0.75$, under-predictions receive three times the penalty of over-predictions of the same magnitude. This is the principal asymmetric setting examined for the transplant-size motivation.

The Thesis studies ordinary quantile regression and Lasso-penalized quantile regression. Three R package implementations are compared:

-   `quantreg`
-   `rqPen`
-   `hqreg`

An important conclusion of the implementation comparison is that the optimization procedures matter: statistical inaccuracies in the `quantreg` penalized optimization process are identified, therefore it is advised against trusting that implementation for this task. The `rqPen` and `hqreg` are highlighted as potentially useful alternatives.

## 4. Model Evaluation

Because the dataset is small and the models involve feature and hyperparameter choices, the Thesis emphasizes repeated validation rather than relying on a single train/test split.

The evaluation framework includes:

1.  **Repeated Nested cross-validation** to reduce selection bias when tuning hyperparameters, provide unbiased prediction error estimates and quantify the uncertainty surrounding them
3.  **Permutation tests** to assess whether two model error distributions differ.
4.  **Bootstrap confidence intervals** to quantify uncertainty in coefficient estimates; the application includes 599 bootstrap samples for the quantile-regression coefficient analysis.

Multiple error criteria are used because no single metric fairly represents every model's objective:

-   Mean squared error (MSE), a symmetric criterion.
-   Mean quantile loss at $\tau=0.5$, a symmetric absolute-error criterion up to scale.
-   Mean quantile loss at $\tau=0.75$, which penalizes under-prediction more heavily.

This distinction is important for interpreting the results: a model optimized for symmetric error may not be the best model under an asymmetric evaluation criterion.

## 5. Main Findings

The conclusions depend on the evaluation objective and are presented thoroughly in the Thesis. A brief summary is discussed here:

Under symmetric criteria, Ridge and Lasso regression perform better than the quantile-regression candidates when assessed using MSE and median ($\tau=0.5$) quantile loss. Ridge performs better than Lasso in this application. This is likely due to the feature-selection stage: after VIF-based filtering, many of the remaining predictors are expected to be informative, a setting in which Ridge can have an advantage over Lasso.

Under asymmetric criteria, specifically under $\tau=0.75$ quantile loss, quantile regression in the upper quantiles becomes more competitive. When fitted at $\tau=0.75$, quantile methods outperform Ridge and Lasso. The penalized `hqreg` and `quantreg` implementations achieve the lowest reported mean $\rho\_{0.75}$ errors, with `hqreg` being slightly better.

However, it must be noted that the differences between higher-quantile regression and Ridge/Lasso under the asymmetric metric are relatively small. In contrast, Ridge and Lasso have a clearer advantage under symmetric metrics. Thus, choosing a model requires deciding which prediction errors matter most for the application.

### Interpretation and uncertainty

Quantile-regression coefficient paths across quantiles can show how predictor associations change across the conditional heart-volume distribution. Prediction intervals formed from multiple quantiles can also communicate a range of plausible heart volumes rather than a single point estimate. These results are shown in detail in the Thesis PDF that this report accompanies.

Bootstrap results were used to examine coefficient variability; the Thesis highlights `hqreg` as providing useful coefficient uncertainty behaviour, with `Ht²` and `Wt²` identified as influential at the 0.75 quantile in the reported analysis.


## 6. Limitations and Responsible Use

The application is based on only 54 pediatric observations after exclusions. This small sample limits the strength and generalizability of the conclusions and makes validation variability and coefficient uncertainty especially important.

The findings are specific to the dataset, feature construction, validation design and model implementations described in the Thesis. They do not establish clinical safety, causal relationships, or suitability for real-world transplant allocation. A prediction of heart volume is only one possible input to donor-recipient compatibility assessment. Clinical use would require independent validation on larger and more representative data, careful assessment of calibration and uncertainty, review by relevant clinical experts, and appropriate governance.

## Reference

Anastasakis, Panagiotis (2023). *Predictive Modeling with Asymmetric Loss Functions*. Diploma Thesis, Department of Mathematics & Applied Mathematics, University of Crete. Supervisor: Dr. Yiannis Kamarianakis. June 16th, 2023.

An extended version of the motivating dataset is discussed in:

Plasencia, J. D., Kamarianakis, Y., Ryan, J. R., et al. (2018). "Alternative methods for virtual heart transplant---Size matching for pediatric heart transplantation with and without donor medical images available." *Pediatric Transplantation*, 22, e13290. https://doi.org/10.1111/petr.13290




## Repository Structure

```text
Heart-Volume-Prediction/
├── Code/
│   ├── Data_exploration_preprocessing.R
│   ├── Main_script.R
│   └── helper.R
├── .DS_Store
├── .gitignore
├── BSc_Thesis.Rproj
├── README.md
└── Thesis.pdf
```

* `Data_exploration_preprocessing.R`: Contains the code for the initial exploration and preprocessing of the pediatric heart-volume dataset, including the preparation of variables for the statistical analysis.
* `Main_script.R`: Contains the main analysis pipeline, including the implementation and comparison of regression models with symmetric and asymmetric loss functions, as well as their evaluation.
* `helper.R`: Contains additional helper functions used throughout the analysis.
* `BSc_Thesis.Rproj`: The RStudio project file, which can be used to open the project in RStudio.
* `.gitignore`: Specifies files and directories that Git should ignore.
* `.DS_Store`: A macOS system file that stores folder display preferences.
* `README.md`: Provides an overview of the Thesis, its methodology, repository contents, and data availability.
* `Thesis.pdf`: Contains the full Thesis, including the theoretical background, methodology, experimental results, and conclusions.

**Data Availability and Reproducibility:** The original dataset used in this study is not publicly distributed and is therefore not included in this repository. Consequently, although the R scripts and analysis procedures are publicly available, the code cannot be executed independently or the reported results fully reproduced without access to the original dataset. Access to the data is subject to the relevant permissions and data-sharing restrictions.
