# lm_for_pred_ab_developability2
This section aims to model antibody aggregation with data from [Jain et al 2017 PNAS](https://www.pnas.org/doi/10.1073/pnas.1616408114), using ESM2 and IgBERT with fine tuning or classical machine learning algorithms.

## Use ESM2 and IgBERT embeddings with trained regression head with or without LoRA fine tuning
![image](https://github.com/user-attachments/assets/8f870ea0-6994-4f91-ae66-b842a2540d1e)

With only ~140 data points, the variability between cross validation folds is evident. Larger backbone (up to 150M) has richer latent space, enabling better performance (ρ = 0.387) than smaller models. ESM2-650M may over fit with much larger feature/sample ratio. LoRA onlys works for small model (8M). 

## Use ESM2 and IgBERT embeddings (frozen) with various ML algorithms
![image](https://github.com/user-attachments/assets/b2b973a0-24da-47c4-a9f1-cf5fe6981b93)

The best-performing solution here is ESM2-150M + ElasticNet (ρ = 0.468), or ESM2-150M + Ridge (ρ = 0.466). IgBERT is not impressive with top algorithm being SVR_linear (ρ = 0.349). Aggregation is considered to be general biophysical property and may not benefit from antibody specific priors in IgBERT.
