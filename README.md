# TSC-of-Supraglacial-Lakes-Evolution-over-GrIS
Time Series Classification of Supraglacial Lakes Evolution over Greenland Ice Sheet


The Greenland Ice Sheet (GrIS) has emerged as a significant contributor to global sea level rise, primarily due to increased meltwater runoff. Supraglacial lakes, which form on the ice sheet surface during the summer months, can impact ice sheet dynamics and mass loss; thus, better understanding these features' seasonal evolution and dynamics is an important task. This study presents a computationally efficient time series classification approach that combines Reconstructed Phase Space (RPS) and Gaussian Mixture Model (GMM) to categorize supraglacial lakes based on their seasonal evolution: those that refreeze at the end of the melt season, those that drain during the melt season, and those that become buried, remaining liquid insulated a few meters beneath the surface. Our approach uses time series data from the Sentinel-1 and Sentinel-2 satellites, which utilize microwave and visible radiation, respectively. Evaluated on a GrIS-wide dataset, the RPS-GMM model, trained on a single representative sample per class, achieves 85.46\% accuracy with Sentinel-1 data alone and 89.70\% with combined Sentinel-1 and Sentinel-2 data. This performance significantly surpasses existing machine learning and deep learning models which require a large training data. The results demonstrate the robustness of the RPS-GMM model in capturing the complex temporal dynamics of supraglacial lakes with minimal training data.
