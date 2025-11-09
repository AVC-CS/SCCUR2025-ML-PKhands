Research Workflow Summary

1. Baseline Problem: Imbalanced Dataset
- Original UCI Poker Hand dataset is ~90% Class 0 & 1, very few samples from rare classes.
- Goal: Test multiple ML models (KNN, Random Forest, Gradient Boosting, XGBoost) to find the best classifier.

2. Result of Phase 1 (Imbalanced Training)
- Accuracy reached ~0.64, but macro F1 was very low (~0.15–0.20).
- Confusion matrix showed models predicted only frequent classes.
- Rare classes received zero recall/F1.
- Conclusion: Problem is not model choice but data imbalance.

3. Hypothesis Change
“The issue is not the model — the issue is the data distribution.”

4. Phase 2: Create a Balanced Dataset
- Generated 10,000 synthetic poker hands (1000 per class).
- All 10 ranks equally represented.
- Same 4 models re-tested.

5. Result of Phase 2 (Balanced Training)
KNN: Accuracy ~0.63, Macro F1 ~0.62
Random Forest: Accuracy ~0.85, Macro F1 ~0.84
Gradient Boosting: Accuracy ~0.82–0.84, Macro F1 ~0.83
XGBoost: Accuracy ~0.87+, Macro F1 ~0.86+ (best)

Key point: All models improved dramatically after balancing.

6. Final Findings
- Imbalanced data produces misleading accuracy and broken learning.
- Model performance ranking changes once data is balanced.
- XGBoost performs best overall on balanced dataset.
- Macro F1 is required to evaluate fairness across all classes.
- Data quality matters more than algorithm complexity.

7. What This Research Shows
Before: High accuracy but useless model.
After: Lower accuracy but fair model.
Before: Rare classes ignored.
After: All classes learned.
Before: Accuracy metric lies.
After: Macro F1 reveals truth.
Before: "Which model is best?"
After: "Is the data learnable?"
