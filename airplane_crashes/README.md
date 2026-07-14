# Flight Crash Classification from ADS-B Traces

This project detects crash flights by how anomalous their trajectories look compared to normal flight behavior.

## Data Sources

We combine two sources, since neither is sufficient on its own. The flight trajectories come from adsb.lol (https://github.com/adsblol), which provides raw ADS-B traces from 2023 to 2026 but no information on which flights were involved in accidents. The accident labels come from the Aviation Safety Network (ASN), which lists recorded crashes but no trajectory data.

To build the labeled dataset, each ASN accident is matched to its corresponding adsb.lol trace via the aircraft's icao code and the accident date, yielding the accident flights. The non-accident flights are drawn as a random sample of regular flights from the same adsb.lol data. Together, these form the labeled accident / non-accident dataset used to train and evaluate the classifier.

## Approach

The task is framed as supervised binary classification: each flight is labeled as either accident or non-accident. A k-nearest-neighbors model is trained on both classes, learning to separate crash trajectories from normal ones based on their extracted flight features. At inference, each flight is classified by the majority label of its nearest neighbors in feature space, producing an accident / non-accident prediction.

Since accidents are far rarer than normal flights, the training data is heavily imbalanced. To address this, we apply SMOTE to oversample the accident class during training, which noticeably improved the model's recall on accidents without hurting its precision.

## Preprocessing

The raw ADS-B traces are first cleaned by removing duplicate records, filtering out outliers, and interpolating gaps to obtain smooth, continuous trajectories. From these cleaned signals we derive motion-based features that capture the aircraft's dynamics like vertical acceleration, acceleration, and curvature. In addition, four frequency-domain features are extracted from the trajectories via FFT: spectral energy, dominant frequency, spectral entropy, and frequency ratio. Together, these time- and frequency-domain features form the input to the classifier.

## Results

On the held-out test set, the KNN classifier reaches an overall accuracy of 98%. It identifies normal flights almost perfectly (precision and recall of 0.99), and for the much rarer accident class it achieves a precision of 0.90 and a recall of 0.81 (F1 = 0.85). In concrete terms, out of 32 real accidents the model correctly flags 26 and misses 6, while raising only 3 false alarms across 400 normal flights. Given the strong class imbalance, this is a solid result. The model catches most accidents while keeping false positives low.

## Repository Structure

The notebooks are ordered by a leading number that reflects the pipeline sequence. Simply go by ascending order (1, 2, 3, ...), as each step builds on the output of the previous one.

## Video

A narrated walkthrough of the project is available on YouTube: [VIDEO_LINK_HERE]
