- [ ] Plot rank correlation of RNDPermut
- [ ] Investigate large action space
- [x] Should focus on static datasets for the mlcad benchs.
- [ ] show that fitting gcn is more costly than mlp for equiv perf.
- [ ] seems like better estimate of V are hard to predict
- [ ] plot with more mpnn layers
- [ ] add remaining benchmarks
- [ ] do a plot showing learned policies are non trivial
## When are MPNNs useful for logic optimization with reinforcement learning?
#### Observation: when optimizing AIGs with a restricted set of primitives, like in most literature on ML for logic optimization, RL does not benefit from graph-aware representations.
- Result: predicting the (near)-optimal value of an AIG does not require MPNN expressivity in theory. 
- Support: LB MSE on MLCAD benchmark and ranking correlations.
- Why it matters: compute time is reduced when not using MPNNs.