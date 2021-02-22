# Federated Recommender System
## Official Title: An Aggregator for Recommendation Engines

### Author: Max Kirker Burton
### Supervisor: Nikos Ntarmos

Recommendation engines are becoming more abundant, with even non-technology sectors now using them to improve sales and user experience. It is important to develop new methods that will improve recommendations that apply to real-life systems. One such method is an aggregation (or a federation), which involves merging recommender algorithms. In this work, we explore two problems faced when implementing a federator: the Same Algorithm Different Dataset (SADD) problem, and the Different Algorithm Same Dataset (DASD) problem. We implement solutions to these problems and compare their effectiveness with popular existing algorithms and baselines. The SADD problem simulates a real-life example of movie providers, and examines whether better recommendations can be made when combining their results. The DASD problem uses two popular algorithms and attempts to federate them to form a single result, judged against each respective algorithms’ performances. Our best solutions for the SADD problem gives an increased performance of 28% and 27% on average compared across baselines. Our best solutions for the DASD problem performed better on precision@k compared to target baselines, and at least as good on NDCG@k.

how to solve this problem?
Traceback (most recent call last):
  File "F:/PyCharm/federator-draft/alg_metrics.py", line 65, in <module>
    TestAlgorithmMetrics(norm_func)
  File "F:/PyCharm/federator-draft/alg_metrics.py", line 47, in __init__
    all_user_precisions, _ = helpers.svd_precision_recall_at_k(svd.predictions, k=k)
AttributeError: 'SurpriseSVD' object has no attribute 'predictions'
