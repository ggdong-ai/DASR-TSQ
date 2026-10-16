# Step 28 formal end-to-end analysis

- Protocol SHA256: `48974a63e13986423f337ab1b2a8f34b07c8e532f5b582453495272058125da6`
- Runs: 30 (6 variants × 5 seeds)
- DASR main directional support: **True**
- Complete method directional support: **True**
- TSQ pressure present: **True**
- Total legitimate early terminations: **218**
- Overall early-termination rate: **0.000727**

## Oracle-budget interpretation

The local budget is a maximum of 101 unique oracle calls. Searches may terminate early only when the local chemical search space is exhausted or remains stagnant for the prespecified limit. All AUC calculations use actual cumulative oracle requests.

## Statistical interpretation

Five paired seeds provide effect-size and direction-consistency evidence. They do not permit a conventional exact two-sided p<0.05 claim.
Step 27 remains the primary high-powered online fixed-budget analysis.

## Multiobjective analysis

The reported four-dimensional hypervolume uses normalized GSK3β activity, JNK3 activity, QED, and SA reward with reference point (0,0,0,0).