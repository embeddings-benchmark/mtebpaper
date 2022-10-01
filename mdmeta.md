---
model-index:
- name: SGPT-5.8B-weightedmean-msmarco-specb-bitfit
  results:
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_counterfactual
      name: MTEB AmazonCounterfactualClassification (en)
      config: en
      split: test
    metrics:
    - type: accuracy
      value: 69.22388059701493
    - type: ap
      value: 32.04724673950256
    - type: f1
      value: 63.25719825770428
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_polarity
      name: MTEB AmazonPolarityClassification
      config: default
      split: test
    metrics:
    - type: accuracy
      value: 71.26109999999998
    - type: ap
      value: 66.16336378255403
    - type: f1
      value: 70.89719145825303
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_reviews_multi
      name: MTEB AmazonReviewsClassification (en)
      config: en
      split: test
    metrics:
    - type: accuracy
      value: 39.19199999999999
    - type: f1
      value: 38.580766731113826
  - task:
      type: Retrieval
    dataset:
      type: arguana
      name: MTEB ArguAna
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 27.311999999999998
    - type: map_at_10
      value: 42.620000000000005
    - type: map_at_100
      value: 43.707
    - type: map_at_1000
      value: 43.714999999999996
    - type: map_at_3
      value: 37.624
    - type: map_at_5
      value: 40.498
    - type: mrr_at_1
      value: 27.667
    - type: mrr_at_10
      value: 42.737
    - type: mrr_at_100
      value: 43.823
    - type: mrr_at_1000
      value: 43.830999999999996
    - type: mrr_at_3
      value: 37.743
    - type: mrr_at_5
      value: 40.616
    - type: ndcg_at_1
      value: 27.311999999999998
    - type: ndcg_at_10
      value: 51.37500000000001
    - type: ndcg_at_100
      value: 55.778000000000006
    - type: ndcg_at_1000
      value: 55.96600000000001
    - type: ndcg_at_3
      value: 41.087
    - type: ndcg_at_5
      value: 46.269
    - type: precision_at_1
      value: 27.311999999999998
    - type: precision_at_10
      value: 7.945
    - type: precision_at_100
      value: 0.9820000000000001
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 17.046
    - type: precision_at_5
      value: 12.745000000000001
    - type: recall_at_1
      value: 27.311999999999998
    - type: recall_at_10
      value: 79.445
    - type: recall_at_100
      value: 98.151
    - type: recall_at_1000
      value: 99.57300000000001
    - type: recall_at_3
      value: 51.13799999999999
    - type: recall_at_5
      value: 63.727000000000004
  - task:
      type: Clustering
    dataset:
      type: mteb/arxiv-clustering-p2p
      name: MTEB ArxivClusteringP2P
      config: default
      split: test
    metrics:
    - type: v_measure
      value: 45.59037428592033
  - task:
      type: Clustering
    dataset:
      type: mteb/arxiv-clustering-s2s
      name: MTEB ArxivClusteringS2S
      config: default
      split: test
    metrics:
    - type: v_measure
      value: 38.86371701986363
  - task:
      type: Reranking
    dataset:
      type: mteb/askubuntudupquestions-reranking
      name: MTEB AskUbuntuDupQuestions
      config: default
      split: test
    metrics:
    - type: map
      value: 61.625568691427766
    - type: mrr
      value: 75.83256386580486
  - task:
      type: STS
    dataset:
      type: mteb/biosses-sts
      name: MTEB BIOSSES
      config: default
      split: test
    metrics:
    - type: cos_sim_pearson
      value: 89.96074355094802
    - type: cos_sim_spearman
      value: 86.2501580394454
    - type: euclidean_pearson
      value: 82.18427440380462
    - type: euclidean_spearman
      value: 80.14760935017947
    - type: manhattan_pearson
      value: 82.24621578156392
    - type: manhattan_spearman
      value: 80.00363016590163
  - task:
      type: Classification
    dataset:
      type: mteb/banking77
      name: MTEB Banking77Classification
      config: default
      split: test
    metrics:
    - type: accuracy
      value: 84.49350649350649
    - type: f1
      value: 84.4249343233736
  - task:
      type: Clustering
    dataset:
      type: mteb/biorxiv-clustering-p2p
      name: MTEB BiorxivClusteringP2P
      config: default
      split: test
    metrics:
    - type: v_measure
      value: 36.551459722989385
  - task:
      type: Clustering
    dataset:
      type: mteb/biorxiv-clustering-s2s
      name: MTEB BiorxivClusteringS2S
      config: default
      split: test
    metrics:
    - type: v_measure
      value: 33.69901851846774
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackAndroidRetrieval
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 30.499
    - type: map_at_10
      value: 41.208
    - type: map_at_100
      value: 42.638
    - type: map_at_1000
      value: 42.754
    - type: map_at_3
      value: 37.506
    - type: map_at_5
      value: 39.422000000000004
    - type: mrr_at_1
      value: 37.339
    - type: mrr_at_10
      value: 47.051
    - type: mrr_at_100
      value: 47.745
    - type: mrr_at_1000
      value: 47.786
    - type: mrr_at_3
      value: 44.086999999999996
    - type: mrr_at_5
      value: 45.711
    - type: ndcg_at_1
      value: 37.339
    - type: ndcg_at_10
      value: 47.666
    - type: ndcg_at_100
      value: 52.994
    - type: ndcg_at_1000
      value: 54.928999999999995
    - type: ndcg_at_3
      value: 41.982
    - type: ndcg_at_5
      value: 44.42
    - type: precision_at_1
      value: 37.339
    - type: precision_at_10
      value: 9.127
    - type: precision_at_100
      value: 1.4749999999999999
    - type: precision_at_1000
      value: 0.194
    - type: precision_at_3
      value: 20.076
    - type: precision_at_5
      value: 14.449000000000002
    - type: recall_at_1
      value: 30.499
    - type: recall_at_10
      value: 60.328
    - type: recall_at_100
      value: 82.57900000000001
    - type: recall_at_1000
      value: 95.074
    - type: recall_at_3
      value: 44.17
    - type: recall_at_5
      value: 50.94
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackEnglishRetrieval
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 30.613
    - type: map_at_10
      value: 40.781
    - type: map_at_100
      value: 42.018
    - type: map_at_1000
      value: 42.132999999999996
    - type: map_at_3
      value: 37.816
    - type: map_at_5
      value: 39.389
    - type: mrr_at_1
      value: 38.408
    - type: mrr_at_10
      value: 46.631
    - type: mrr_at_100
      value: 47.332
    - type: mrr_at_1000
      value: 47.368
    - type: mrr_at_3
      value: 44.384
    - type: mrr_at_5
      value: 45.661
    - type: ndcg_at_1
      value: 38.408
    - type: ndcg_at_10
      value: 46.379999999999995
    - type: ndcg_at_100
      value: 50.81
    - type: ndcg_at_1000
      value: 52.663000000000004
    - type: ndcg_at_3
      value: 42.18
    - type: ndcg_at_5
      value: 43.974000000000004
    - type: precision_at_1
      value: 38.408
    - type: precision_at_10
      value: 8.656
    - type: precision_at_100
      value: 1.3860000000000001
    - type: precision_at_1000
      value: 0.184
    - type: precision_at_3
      value: 20.276
    - type: precision_at_5
      value: 14.241999999999999
    - type: recall_at_1
      value: 30.613
    - type: recall_at_10
      value: 56.44
    - type: recall_at_100
      value: 75.044
    - type: recall_at_1000
      value: 86.426
    - type: recall_at_3
      value: 43.766
    - type: recall_at_5
      value: 48.998000000000005
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackGamingRetrieval
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 37.370999999999995
    - type: map_at_10
      value: 49.718
    - type: map_at_100
      value: 50.737
    - type: map_at_1000
      value: 50.79
    - type: map_at_3
      value: 46.231
    - type: map_at_5
      value: 48.329
    - type: mrr_at_1
      value: 42.884
    - type: mrr_at_10
      value: 53.176
    - type: mrr_at_100
      value: 53.81700000000001
    - type: mrr_at_1000
      value: 53.845
    - type: mrr_at_3
      value: 50.199000000000005
    - type: mrr_at_5
      value: 52.129999999999995
    - type: ndcg_at_1
      value: 42.884
    - type: ndcg_at_10
      value: 55.826
    - type: ndcg_at_100
      value: 59.93000000000001
    - type: ndcg_at_1000
      value: 61.013
    - type: ndcg_at_3
      value: 49.764
    - type: ndcg_at_5
      value: 53.025999999999996
    - type: precision_at_1
      value: 42.884
    - type: precision_at_10
      value: 9.046999999999999
    - type: precision_at_100
      value: 1.212
    - type: precision_at_1000
      value: 0.135
    - type: precision_at_3
      value: 22.131999999999998
    - type: precision_at_5
      value: 15.524
    - type: recall_at_1
      value: 37.370999999999995
    - type: recall_at_10
      value: 70.482
    - type: recall_at_100
      value: 88.425
    - type: recall_at_1000
      value: 96.03399999999999
    - type: recall_at_3
      value: 54.43
    - type: recall_at_5
      value: 62.327999999999996
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackGisRetrieval
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 22.875999999999998
    - type: map_at_10
      value: 31.715
    - type: map_at_100
      value: 32.847
    - type: map_at_1000
      value: 32.922000000000004
    - type: map_at_3
      value: 29.049999999999997
    - type: map_at_5
      value: 30.396
    - type: mrr_at_1
      value: 24.52
    - type: mrr_at_10
      value: 33.497
    - type: mrr_at_100
      value: 34.455000000000005
    - type: mrr_at_1000
      value: 34.510000000000005
    - type: mrr_at_3
      value: 30.791
    - type: mrr_at_5
      value: 32.175
    - type: ndcg_at_1
      value: 24.52
    - type: ndcg_at_10
      value: 36.95
    - type: ndcg_at_100
      value: 42.238
    - type: ndcg_at_1000
      value: 44.147999999999996
    - type: ndcg_at_3
      value: 31.435000000000002
    - type: ndcg_at_5
      value: 33.839000000000006
    - type: precision_at_1
      value: 24.52
    - type: precision_at_10
      value: 5.9319999999999995
    - type: precision_at_100
      value: 0.901
    - type: precision_at_1000
      value: 0.11
    - type: precision_at_3
      value: 13.446
    - type: precision_at_5
      value: 9.469
    - type: recall_at_1
      value: 22.875999999999998
    - type: recall_at_10
      value: 51.38
    - type: recall_at_100
      value: 75.31099999999999
    - type: recall_at_1000
      value: 89.718
    - type: recall_at_3
      value: 36.26
    - type: recall_at_5
      value: 42.248999999999995
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackMathematicaRetrieval
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 14.984
    - type: map_at_10
      value: 23.457
    - type: map_at_100
      value: 24.723
    - type: map_at_1000
      value: 24.846
    - type: map_at_3
      value: 20.873
    - type: map_at_5
      value: 22.357
    - type: mrr_at_1
      value: 18.159
    - type: mrr_at_10
      value: 27.431
    - type: mrr_at_100
      value: 28.449
    - type: mrr_at_1000
      value: 28.52
    - type: mrr_at_3
      value: 24.979000000000003
    - type: mrr_at_5
      value: 26.447
    - type: ndcg_at_1
      value: 18.159
    - type: ndcg_at_10
      value: 28.627999999999997
    - type: ndcg_at_100
      value: 34.741
    - type: ndcg_at_1000
      value: 37.516
    - type: ndcg_at_3
      value: 23.902
    - type: ndcg_at_5
      value: 26.294
    - type: precision_at_1
      value: 18.159
    - type: precision_at_10
      value: 5.485
    - type: precision_at_100
      value: 0.985
    - type: precision_at_1000
      value: 0.136
    - type: precision_at_3
      value: 11.774
    - type: precision_at_5
      value: 8.731
    - type: recall_at_1
      value: 14.984
    - type: recall_at_10
      value: 40.198
    - type: recall_at_100
      value: 67.11500000000001
    - type: recall_at_1000
      value: 86.497
    - type: recall_at_3
      value: 27.639000000000003
    - type: recall_at_5
      value: 33.595000000000006
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackPhysicsRetrieval
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 29.067
    - type: map_at_10
      value: 39.457
    - type: map_at_100
      value: 40.83
    - type: map_at_1000
      value: 40.94
    - type: map_at_3
      value: 35.995
    - type: map_at_5
      value: 38.159
    - type: mrr_at_1
      value: 34.937000000000005
    - type: mrr_at_10
      value: 44.755
    - type: mrr_at_100
      value: 45.549
    - type: mrr_at_1000
      value: 45.589
    - type: mrr_at_3
      value: 41.947
    - type: mrr_at_5
      value: 43.733
    - type: ndcg_at_1
      value: 34.937000000000005
    - type: ndcg_at_10
      value: 45.573
    - type: ndcg_at_100
      value: 51.266999999999996
    - type: ndcg_at_1000
      value: 53.184
    - type: ndcg_at_3
      value: 39.961999999999996
    - type: ndcg_at_5
      value: 43.02
    - type: precision_at_1
      value: 34.937000000000005
    - type: precision_at_10
      value: 8.296000000000001
    - type: precision_at_100
      value: 1.32
    - type: precision_at_1000
      value: 0.167
    - type: precision_at_3
      value: 18.8
    - type: precision_at_5
      value: 13.763
    - type: recall_at_1
      value: 29.067
    - type: recall_at_10
      value: 58.298
    - type: recall_at_100
      value: 82.25099999999999
    - type: recall_at_1000
      value: 94.476
    - type: recall_at_3
      value: 42.984
    - type: recall_at_5
      value: 50.658
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackProgrammersRetrieval
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 25.985999999999997
    - type: map_at_10
      value: 35.746
    - type: map_at_100
      value: 37.067
    - type: map_at_1000
      value: 37.191
    - type: map_at_3
      value: 32.599000000000004
    - type: map_at_5
      value: 34.239000000000004
    - type: mrr_at_1
      value: 31.735000000000003
    - type: mrr_at_10
      value: 40.515
    - type: mrr_at_100
      value: 41.459
    - type: mrr_at_1000
      value: 41.516
    - type: mrr_at_3
      value: 37.938
    - type: mrr_at_5
      value: 39.25
    - type: ndcg_at_1
      value: 31.735000000000003
    - type: ndcg_at_10
      value: 41.484
    - type: ndcg_at_100
      value: 47.047
    - type: ndcg_at_1000
      value: 49.427
    - type: ndcg_at_3
      value: 36.254999999999995
    - type: ndcg_at_5
      value: 38.375
    - type: precision_at_1
      value: 31.735000000000003
    - type: precision_at_10
      value: 7.66
    - type: precision_at_100
      value: 1.234
    - type: precision_at_1000
      value: 0.16
    - type: precision_at_3
      value: 17.427999999999997
    - type: precision_at_5
      value: 12.328999999999999
    - type: recall_at_1
      value: 25.985999999999997
    - type: recall_at_10
      value: 53.761
    - type: recall_at_100
      value: 77.149
    - type: recall_at_1000
      value: 93.342
    - type: recall_at_3
      value: 39.068000000000005
    - type: recall_at_5
      value: 44.693
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackRetrieval
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 24.949749999999998
    - type: map_at_10
      value: 34.04991666666667
    - type: map_at_100
      value: 35.26825
    - type: map_at_1000
      value: 35.38316666666667
    - type: map_at_3
      value: 31.181333333333335
    - type: map_at_5
      value: 32.77391666666667
    - type: mrr_at_1
      value: 29.402833333333334
    - type: mrr_at_10
      value: 38.01633333333333
    - type: mrr_at_100
      value: 38.88033333333334
    - type: mrr_at_1000
      value: 38.938500000000005
    - type: mrr_at_3
      value: 35.5175
    - type: mrr_at_5
      value: 36.93808333333333
    - type: ndcg_at_1
      value: 29.402833333333334
    - type: ndcg_at_10
      value: 39.403166666666664
    - type: ndcg_at_100
      value: 44.66408333333333
    - type: ndcg_at_1000
      value: 46.96283333333333
    - type: ndcg_at_3
      value: 34.46633333333334
    - type: ndcg_at_5
      value: 36.78441666666667
    - type: precision_at_1
      value: 29.402833333333334
    - type: precision_at_10
      value: 6.965833333333333
    - type: precision_at_100
      value: 1.1330833333333334
    - type: precision_at_1000
      value: 0.15158333333333335
    - type: precision_at_3
      value: 15.886666666666665
    - type: precision_at_5
      value: 11.360416666666667
    - type: recall_at_1
      value: 24.949749999999998
    - type: recall_at_10
      value: 51.29325
    - type: recall_at_100
      value: 74.3695
    - type: recall_at_1000
      value: 90.31299999999999
    - type: recall_at_3
      value: 37.580083333333334
    - type: recall_at_5
      value: 43.529666666666664
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackStatsRetrieval
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 22.081999999999997
    - type: map_at_10
      value: 29.215999999999998
    - type: map_at_100
      value: 30.163
    - type: map_at_1000
      value: 30.269000000000002
    - type: map_at_3
      value: 26.942
    - type: map_at_5
      value: 28.236
    - type: mrr_at_1
      value: 24.847
    - type: mrr_at_10
      value: 31.918999999999997
    - type: mrr_at_100
      value: 32.817
    - type: mrr_at_1000
      value: 32.897
    - type: mrr_at_3
      value: 29.831000000000003
    - type: mrr_at_5
      value: 31.019999999999996
    - type: ndcg_at_1
      value: 24.847
    - type: ndcg_at_10
      value: 33.4
    - type: ndcg_at_100
      value: 38.354
    - type: ndcg_at_1000
      value: 41.045
    - type: ndcg_at_3
      value: 29.236
    - type: ndcg_at_5
      value: 31.258000000000003
    - type: precision_at_1
      value: 24.847
    - type: precision_at_10
      value: 5.353
    - type: precision_at_100
      value: 0.853
    - type: precision_at_1000
      value: 0.116
    - type: precision_at_3
      value: 12.679000000000002
    - type: precision_at_5
      value: 8.988
    - type: recall_at_1
      value: 22.081999999999997
    - type: recall_at_10
      value: 43.505
    - type: recall_at_100
      value: 66.45400000000001
    - type: recall_at_1000
      value: 86.378
    - type: recall_at_3
      value: 32.163000000000004
    - type: recall_at_5
      value: 37.059999999999995
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackTexRetrieval
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 15.540000000000001
    - type: map_at_10
      value: 22.362000000000002
    - type: map_at_100
      value: 23.435
    - type: map_at_1000
      value: 23.564
    - type: map_at_3
      value: 20.143
    - type: map_at_5
      value: 21.324
    - type: mrr_at_1
      value: 18.892
    - type: mrr_at_10
      value: 25.942999999999998
    - type: mrr_at_100
      value: 26.883000000000003
    - type: mrr_at_1000
      value: 26.968999999999998
    - type: mrr_at_3
      value: 23.727
    - type: mrr_at_5
      value: 24.923000000000002
    - type: ndcg_at_1
      value: 18.892
    - type: ndcg_at_10
      value: 26.811
    - type: ndcg_at_100
      value: 32.066
    - type: ndcg_at_1000
      value: 35.166
    - type: ndcg_at_3
      value: 22.706
    - type: ndcg_at_5
      value: 24.508
    - type: precision_at_1
      value: 18.892
    - type: precision_at_10
      value: 4.942
    - type: precision_at_100
      value: 0.878
    - type: precision_at_1000
      value: 0.131
    - type: precision_at_3
      value: 10.748000000000001
    - type: precision_at_5
      value: 7.784000000000001
    - type: recall_at_1
      value: 15.540000000000001
    - type: recall_at_10
      value: 36.742999999999995
    - type: recall_at_100
      value: 60.525
    - type: recall_at_1000
      value: 82.57600000000001
    - type: recall_at_3
      value: 25.252000000000002
    - type: recall_at_5
      value: 29.872
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackUnixRetrieval
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 24.453
    - type: map_at_10
      value: 33.363
    - type: map_at_100
      value: 34.579
    - type: map_at_1000
      value: 34.686
    - type: map_at_3
      value: 30.583
    - type: map_at_5
      value: 32.118
    - type: mrr_at_1
      value: 28.918
    - type: mrr_at_10
      value: 37.675
    - type: mrr_at_100
      value: 38.567
    - type: mrr_at_1000
      value: 38.632
    - type: mrr_at_3
      value: 35.260999999999996
    - type: mrr_at_5
      value: 36.576
    - type: ndcg_at_1
      value: 28.918
    - type: ndcg_at_10
      value: 38.736
    - type: ndcg_at_100
      value: 44.261
    - type: ndcg_at_1000
      value: 46.72
    - type: ndcg_at_3
      value: 33.81
    - type: ndcg_at_5
      value: 36.009
    - type: precision_at_1
      value: 28.918
    - type: precision_at_10
      value: 6.586
    - type: precision_at_100
      value: 1.047
    - type: precision_at_1000
      value: 0.13699999999999998
    - type: precision_at_3
      value: 15.360999999999999
    - type: precision_at_5
      value: 10.857999999999999
    - type: recall_at_1
      value: 24.453
    - type: recall_at_10
      value: 50.885999999999996
    - type: recall_at_100
      value: 75.03
    - type: recall_at_1000
      value: 92.123
    - type: recall_at_3
      value: 37.138
    - type: recall_at_5
      value: 42.864999999999995
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackWebmastersRetrieval
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 24.57
    - type: map_at_10
      value: 33.672000000000004
    - type: map_at_100
      value: 35.244
    - type: map_at_1000
      value: 35.467
    - type: map_at_3
      value: 30.712
    - type: map_at_5
      value: 32.383
    - type: mrr_at_1
      value: 29.644
    - type: mrr_at_10
      value: 38.344
    - type: mrr_at_100
      value: 39.219
    - type: mrr_at_1000
      value: 39.282000000000004
    - type: mrr_at_3
      value: 35.771
    - type: mrr_at_5
      value: 37.273
    - type: ndcg_at_1
      value: 29.644
    - type: ndcg_at_10
      value: 39.567
    - type: ndcg_at_100
      value: 45.097
    - type: ndcg_at_1000
      value: 47.923
    - type: ndcg_at_3
      value: 34.768
    - type: ndcg_at_5
      value: 37.122
    - type: precision_at_1
      value: 29.644
    - type: precision_at_10
      value: 7.5889999999999995
    - type: precision_at_100
      value: 1.478
    - type: precision_at_1000
      value: 0.23500000000000001
    - type: precision_at_3
      value: 16.337
    - type: precision_at_5
      value: 12.055
    - type: recall_at_1
      value: 24.57
    - type: recall_at_10
      value: 51.00900000000001
    - type: recall_at_100
      value: 75.423
    - type: recall_at_1000
      value: 93.671
    - type: recall_at_3
      value: 36.925999999999995
    - type: recall_at_5
      value: 43.245
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackWordpressRetrieval
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 21.356
    - type: map_at_10
      value: 27.904
    - type: map_at_100
      value: 28.938000000000002
    - type: map_at_1000
      value: 29.036
    - type: map_at_3
      value: 25.726
    - type: map_at_5
      value: 26.935
    - type: mrr_at_1
      value: 22.551
    - type: mrr_at_10
      value: 29.259
    - type: mrr_at_100
      value: 30.272
    - type: mrr_at_1000
      value: 30.348000000000003
    - type: mrr_at_3
      value: 27.295
    - type: mrr_at_5
      value: 28.358
    - type: ndcg_at_1
      value: 22.551
    - type: ndcg_at_10
      value: 31.817
    - type: ndcg_at_100
      value: 37.164
    - type: ndcg_at_1000
      value: 39.82
    - type: ndcg_at_3
      value: 27.595999999999997
    - type: ndcg_at_5
      value: 29.568
    - type: precision_at_1
      value: 22.551
    - type: precision_at_10
      value: 4.917
    - type: precision_at_100
      value: 0.828
    - type: precision_at_1000
      value: 0.11399999999999999
    - type: precision_at_3
      value: 11.583
    - type: precision_at_5
      value: 8.133
    - type: recall_at_1
      value: 21.356
    - type: recall_at_10
      value: 42.489
    - type: recall_at_100
      value: 67.128
    - type: recall_at_1000
      value: 87.441
    - type: recall_at_3
      value: 31.165
    - type: recall_at_5
      value: 35.853
  - task:
      type: Retrieval
    dataset:
      type: climate-fever
      name: MTEB ClimateFEVER
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 12.306000000000001
    - type: map_at_10
      value: 21.523
    - type: map_at_100
      value: 23.358
    - type: map_at_1000
      value: 23.541
    - type: map_at_3
      value: 17.809
    - type: map_at_5
      value: 19.631
    - type: mrr_at_1
      value: 27.948
    - type: mrr_at_10
      value: 40.355000000000004
    - type: mrr_at_100
      value: 41.166000000000004
    - type: mrr_at_1000
      value: 41.203
    - type: mrr_at_3
      value: 36.819
    - type: mrr_at_5
      value: 38.958999999999996
    - type: ndcg_at_1
      value: 27.948
    - type: ndcg_at_10
      value: 30.462
    - type: ndcg_at_100
      value: 37.473
    - type: ndcg_at_1000
      value: 40.717999999999996
    - type: ndcg_at_3
      value: 24.646
    - type: ndcg_at_5
      value: 26.642
    - type: precision_at_1
      value: 27.948
    - type: precision_at_10
      value: 9.648
    - type: precision_at_100
      value: 1.7239999999999998
    - type: precision_at_1000
      value: 0.232
    - type: precision_at_3
      value: 18.48
    - type: precision_at_5
      value: 14.293
    - type: recall_at_1
      value: 12.306000000000001
    - type: recall_at_10
      value: 37.181
    - type: recall_at_100
      value: 61.148
    - type: recall_at_1000
      value: 79.401
    - type: recall_at_3
      value: 22.883
    - type: recall_at_5
      value: 28.59
  - task:
      type: Retrieval
    dataset:
      type: dbpedia-entity
      name: MTEB DBPedia
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 9.357
    - type: map_at_10
      value: 18.849
    - type: map_at_100
      value: 25.369000000000003
    - type: map_at_1000
      value: 26.950000000000003
    - type: map_at_3
      value: 13.625000000000002
    - type: map_at_5
      value: 15.956999999999999
    - type: mrr_at_1
      value: 67.75
    - type: mrr_at_10
      value: 74.734
    - type: mrr_at_100
      value: 75.1
    - type: mrr_at_1000
      value: 75.10900000000001
    - type: mrr_at_3
      value: 73.542
    - type: mrr_at_5
      value: 74.167
    - type: ndcg_at_1
      value: 55.375
    - type: ndcg_at_10
      value: 39.873999999999995
    - type: ndcg_at_100
      value: 43.098
    - type: ndcg_at_1000
      value: 50.69200000000001
    - type: ndcg_at_3
      value: 44.856
    - type: ndcg_at_5
      value: 42.138999999999996
    - type: precision_at_1
      value: 67.75
    - type: precision_at_10
      value: 31.1
    - type: precision_at_100
      value: 9.303
    - type: precision_at_1000
      value: 2.0060000000000002
    - type: precision_at_3
      value: 48.25
    - type: precision_at_5
      value: 40.949999999999996
    - type: recall_at_1
      value: 9.357
    - type: recall_at_10
      value: 23.832
    - type: recall_at_100
      value: 47.906
    - type: recall_at_1000
      value: 71.309
    - type: recall_at_3
      value: 14.512
    - type: recall_at_5
      value: 18.3
  - task:
      type: Classification
    dataset:
      type: mteb/emotion
      name: MTEB EmotionClassification
      config: default
      split: test
    metrics:
    - type: accuracy
      value: 49.655
    - type: f1
      value: 45.51976190938951
  - task:
      type: Retrieval
    dataset:
      type: fever
      name: MTEB FEVER
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 62.739999999999995
    - type: map_at_10
      value: 73.07000000000001
    - type: map_at_100
      value: 73.398
    - type: map_at_1000
      value: 73.41
    - type: map_at_3
      value: 71.33800000000001
    - type: map_at_5
      value: 72.423
    - type: mrr_at_1
      value: 67.777
    - type: mrr_at_10
      value: 77.873
    - type: mrr_at_100
      value: 78.091
    - type: mrr_at_1000
      value: 78.094
    - type: mrr_at_3
      value: 76.375
    - type: mrr_at_5
      value: 77.316
    - type: ndcg_at_1
      value: 67.777
    - type: ndcg_at_10
      value: 78.24
    - type: ndcg_at_100
      value: 79.557
    - type: ndcg_at_1000
      value: 79.814
    - type: ndcg_at_3
      value: 75.125
    - type: ndcg_at_5
      value: 76.834
    - type: precision_at_1
      value: 67.777
    - type: precision_at_10
      value: 9.832
    - type: precision_at_100
      value: 1.061
    - type: precision_at_1000
      value: 0.11
    - type: precision_at_3
      value: 29.433
    - type: precision_at_5
      value: 18.665000000000003
    - type: recall_at_1
      value: 62.739999999999995
    - type: recall_at_10
      value: 89.505
    - type: recall_at_100
      value: 95.102
    - type: recall_at_1000
      value: 96.825
    - type: recall_at_3
      value: 81.028
    - type: recall_at_5
      value: 85.28099999999999
  - task:
      type: Retrieval
    dataset:
      type: fiqa
      name: MTEB FiQA2018
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 18.467
    - type: map_at_10
      value: 30.020999999999997
    - type: map_at_100
      value: 31.739
    - type: map_at_1000
      value: 31.934
    - type: map_at_3
      value: 26.003
    - type: map_at_5
      value: 28.338
    - type: mrr_at_1
      value: 35.339999999999996
    - type: mrr_at_10
      value: 44.108999999999995
    - type: mrr_at_100
      value: 44.993
    - type: mrr_at_1000
      value: 45.042
    - type: mrr_at_3
      value: 41.667
    - type: mrr_at_5
      value: 43.14
    - type: ndcg_at_1
      value: 35.339999999999996
    - type: ndcg_at_10
      value: 37.202
    - type: ndcg_at_100
      value: 43.852999999999994
    - type: ndcg_at_1000
      value: 47.235
    - type: ndcg_at_3
      value: 33.5
    - type: ndcg_at_5
      value: 34.985
    - type: precision_at_1
      value: 35.339999999999996
    - type: precision_at_10
      value: 10.247
    - type: precision_at_100
      value: 1.7149999999999999
    - type: precision_at_1000
      value: 0.232
    - type: precision_at_3
      value: 22.222
    - type: precision_at_5
      value: 16.573999999999998
    - type: recall_at_1
      value: 18.467
    - type: recall_at_10
      value: 44.080999999999996
    - type: recall_at_100
      value: 68.72200000000001
    - type: recall_at_1000
      value: 89.087
    - type: recall_at_3
      value: 30.567
    - type: recall_at_5
      value: 36.982
  - task:
      type: Retrieval
    dataset:
      type: hotpotqa
      name: MTEB HotpotQA
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 35.726
    - type: map_at_10
      value: 50.207
    - type: map_at_100
      value: 51.05499999999999
    - type: map_at_1000
      value: 51.12799999999999
    - type: map_at_3
      value: 47.576
    - type: map_at_5
      value: 49.172
    - type: mrr_at_1
      value: 71.452
    - type: mrr_at_10
      value: 77.41900000000001
    - type: mrr_at_100
      value: 77.711
    - type: mrr_at_1000
      value: 77.723
    - type: mrr_at_3
      value: 76.39399999999999
    - type: mrr_at_5
      value: 77.00099999999999
    - type: ndcg_at_1
      value: 71.452
    - type: ndcg_at_10
      value: 59.260999999999996
    - type: ndcg_at_100
      value: 62.424
    - type: ndcg_at_1000
      value: 63.951
    - type: ndcg_at_3
      value: 55.327000000000005
    - type: ndcg_at_5
      value: 57.416999999999994
    - type: precision_at_1
      value: 71.452
    - type: precision_at_10
      value: 12.061
    - type: precision_at_100
      value: 1.455
    - type: precision_at_1000
      value: 0.166
    - type: precision_at_3
      value: 34.36
    - type: precision_at_5
      value: 22.266
    - type: recall_at_1
      value: 35.726
    - type: recall_at_10
      value: 60.304
    - type: recall_at_100
      value: 72.75500000000001
    - type: recall_at_1000
      value: 82.978
    - type: recall_at_3
      value: 51.54
    - type: recall_at_5
      value: 55.665
  - task:
      type: Classification
    dataset:
      type: mteb/imdb
      name: MTEB ImdbClassification
      config: default
      split: test
    metrics:
    - type: accuracy
      value: 66.63759999999999
    - type: ap
      value: 61.48938261286748
    - type: f1
      value: 66.35089269264965
  - task:
      type: Retrieval
    dataset:
      type: msmarco
      name: MTEB MSMARCO
      config: default
      split: validation
    metrics:
    - type: map_at_1
      value: 20.842
    - type: map_at_10
      value: 32.992
    - type: map_at_100
      value: 34.236
    - type: map_at_1000
      value: 34.286
    - type: map_at_3
      value: 29.049000000000003
    - type: map_at_5
      value: 31.391999999999996
    - type: mrr_at_1
      value: 21.375
    - type: mrr_at_10
      value: 33.581
    - type: mrr_at_100
      value: 34.760000000000005
    - type: mrr_at_1000
      value: 34.803
    - type: mrr_at_3
      value: 29.704000000000004
    - type: mrr_at_5
      value: 32.015
    - type: ndcg_at_1
      value: 21.375
    - type: ndcg_at_10
      value: 39.905
    - type: ndcg_at_100
      value: 45.843
    - type: ndcg_at_1000
      value: 47.083999999999996
    - type: ndcg_at_3
      value: 31.918999999999997
    - type: ndcg_at_5
      value: 36.107
    - type: precision_at_1
      value: 21.375
    - type: precision_at_10
      value: 6.393
    - type: precision_at_100
      value: 0.935
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 13.663
    - type: precision_at_5
      value: 10.324
    - type: recall_at_1
      value: 20.842
    - type: recall_at_10
      value: 61.17
    - type: recall_at_100
      value: 88.518
    - type: recall_at_1000
      value: 97.993
    - type: recall_at_3
      value: 39.571
    - type: recall_at_5
      value: 49.653999999999996
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_domain
      name: MTEB MTOPDomainClassification (en)
      config: en
      split: test
    metrics:
    - type: accuracy
      value: 93.46557227542178
    - type: f1
      value: 92.87345917772146
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_intent
      name: MTEB MTOPIntentClassification (en)
      config: en
      split: test
    metrics:
    - type: accuracy
      value: 72.42134062927497
    - type: f1
      value: 55.03624810959269
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (en)
      config: en
      split: test
    metrics:
    - type: accuracy
      value: 70.3866845998655
    - type: f1
      value: 68.9674519872921
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (en)
      config: en
      split: test
    metrics:
    - type: accuracy
      value: 76.27774041694687
    - type: f1
      value: 76.72936190462792
  - task:
      type: Clustering
    dataset:
      type: mteb/medrxiv-clustering-p2p
      name: MTEB MedrxivClusteringP2P
      config: default
      split: test
    metrics:
    - type: v_measure
      value: 31.511745925773337
  - task:
      type: Clustering
    dataset:
      type: mteb/medrxiv-clustering-s2s
      name: MTEB MedrxivClusteringS2S
      config: default
      split: test
    metrics:
    - type: v_measure
      value: 28.764235987575365
  - task:
      type: Reranking
    dataset:
      type: mteb/mind_small
      name: MTEB MindSmallReranking
      config: default
      split: test
    metrics:
    - type: map
      value: 32.29353136386601
    - type: mrr
      value: 33.536774455851685
  - task:
      type: Retrieval
    dataset:
      type: nfcorpus
      name: MTEB NFCorpus
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 5.702
    - type: map_at_10
      value: 13.642000000000001
    - type: map_at_100
      value: 17.503
    - type: map_at_1000
      value: 19.126
    - type: map_at_3
      value: 9.748
    - type: map_at_5
      value: 11.642
    - type: mrr_at_1
      value: 45.82
    - type: mrr_at_10
      value: 54.821
    - type: mrr_at_100
      value: 55.422000000000004
    - type: mrr_at_1000
      value: 55.452999999999996
    - type: mrr_at_3
      value: 52.373999999999995
    - type: mrr_at_5
      value: 53.937000000000005
    - type: ndcg_at_1
      value: 44.272
    - type: ndcg_at_10
      value: 36.213
    - type: ndcg_at_100
      value: 33.829
    - type: ndcg_at_1000
      value: 42.557
    - type: ndcg_at_3
      value: 40.814
    - type: ndcg_at_5
      value: 39.562000000000005
    - type: precision_at_1
      value: 45.511
    - type: precision_at_10
      value: 27.214
    - type: precision_at_100
      value: 8.941
    - type: precision_at_1000
      value: 2.1870000000000003
    - type: precision_at_3
      value: 37.874
    - type: precision_at_5
      value: 34.489
    - type: recall_at_1
      value: 5.702
    - type: recall_at_10
      value: 17.638
    - type: recall_at_100
      value: 34.419
    - type: recall_at_1000
      value: 66.41
    - type: recall_at_3
      value: 10.914
    - type: recall_at_5
      value: 14.032
  - task:
      type: Retrieval
    dataset:
      type: nq
      name: MTEB NQ
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 30.567
    - type: map_at_10
      value: 45.01
    - type: map_at_100
      value: 46.091
    - type: map_at_1000
      value: 46.126
    - type: map_at_3
      value: 40.897
    - type: map_at_5
      value: 43.301
    - type: mrr_at_1
      value: 34.56
    - type: mrr_at_10
      value: 47.725
    - type: mrr_at_100
      value: 48.548
    - type: mrr_at_1000
      value: 48.571999999999996
    - type: mrr_at_3
      value: 44.361
    - type: mrr_at_5
      value: 46.351
    - type: ndcg_at_1
      value: 34.531
    - type: ndcg_at_10
      value: 52.410000000000004
    - type: ndcg_at_100
      value: 56.999
    - type: ndcg_at_1000
      value: 57.830999999999996
    - type: ndcg_at_3
      value: 44.734
    - type: ndcg_at_5
      value: 48.701
    - type: precision_at_1
      value: 34.531
    - type: precision_at_10
      value: 8.612
    - type: precision_at_100
      value: 1.118
    - type: precision_at_1000
      value: 0.12
    - type: precision_at_3
      value: 20.307
    - type: precision_at_5
      value: 14.519000000000002
    - type: recall_at_1
      value: 30.567
    - type: recall_at_10
      value: 72.238
    - type: recall_at_100
      value: 92.154
    - type: recall_at_1000
      value: 98.375
    - type: recall_at_3
      value: 52.437999999999995
    - type: recall_at_5
      value: 61.516999999999996
  - task:
      type: Retrieval
    dataset:
      type: quora
      name: MTEB QuoraRetrieval
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 65.98
    - type: map_at_10
      value: 80.05600000000001
    - type: map_at_100
      value: 80.76299999999999
    - type: map_at_1000
      value: 80.786
    - type: map_at_3
      value: 76.848
    - type: map_at_5
      value: 78.854
    - type: mrr_at_1
      value: 75.86
    - type: mrr_at_10
      value: 83.397
    - type: mrr_at_100
      value: 83.555
    - type: mrr_at_1000
      value: 83.557
    - type: mrr_at_3
      value: 82.033
    - type: mrr_at_5
      value: 82.97
    - type: ndcg_at_1
      value: 75.88000000000001
    - type: ndcg_at_10
      value: 84.58099999999999
    - type: ndcg_at_100
      value: 86.151
    - type: ndcg_at_1000
      value: 86.315
    - type: ndcg_at_3
      value: 80.902
    - type: ndcg_at_5
      value: 82.953
    - type: precision_at_1
      value: 75.88000000000001
    - type: precision_at_10
      value: 12.986
    - type: precision_at_100
      value: 1.5110000000000001
    - type: precision_at_1000
      value: 0.156
    - type: precision_at_3
      value: 35.382999999999996
    - type: precision_at_5
      value: 23.555999999999997
    - type: recall_at_1
      value: 65.98
    - type: recall_at_10
      value: 93.716
    - type: recall_at_100
      value: 99.21799999999999
    - type: recall_at_1000
      value: 99.97
    - type: recall_at_3
      value: 83.551
    - type: recall_at_5
      value: 88.998
  - task:
      type: Clustering
    dataset:
      type: mteb/reddit-clustering
      name: MTEB RedditClustering
      config: default
      split: test
    metrics:
    - type: v_measure
      value: 40.45148482612238
  - task:
      type: Clustering
    dataset:
      type: mteb/reddit-clustering-p2p
      name: MTEB RedditClusteringP2P
      config: default
      split: test
    metrics:
    - type: v_measure
      value: 55.749490673039126
  - task:
      type: Retrieval
    dataset:
      type: scidocs
      name: MTEB SCIDOCS
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 4.903
    - type: map_at_10
      value: 11.926
    - type: map_at_100
      value: 13.916999999999998
    - type: map_at_1000
      value: 14.215
    - type: map_at_3
      value: 8.799999999999999
    - type: map_at_5
      value: 10.360999999999999
    - type: mrr_at_1
      value: 24.099999999999998
    - type: mrr_at_10
      value: 34.482
    - type: mrr_at_100
      value: 35.565999999999995
    - type: mrr_at_1000
      value: 35.619
    - type: mrr_at_3
      value: 31.433
    - type: mrr_at_5
      value: 33.243
    - type: ndcg_at_1
      value: 24.099999999999998
    - type: ndcg_at_10
      value: 19.872999999999998
    - type: ndcg_at_100
      value: 27.606
    - type: ndcg_at_1000
      value: 32.811
    - type: ndcg_at_3
      value: 19.497999999999998
    - type: ndcg_at_5
      value: 16.813
    - type: precision_at_1
      value: 24.099999999999998
    - type: precision_at_10
      value: 10.08
    - type: precision_at_100
      value: 2.122
    - type: precision_at_1000
      value: 0.337
    - type: precision_at_3
      value: 18.2
    - type: precision_at_5
      value: 14.62
    - type: recall_at_1
      value: 4.903
    - type: recall_at_10
      value: 20.438000000000002
    - type: recall_at_100
      value: 43.043
    - type: recall_at_1000
      value: 68.41000000000001
    - type: recall_at_3
      value: 11.068
    - type: recall_at_5
      value: 14.818000000000001
  - task:
      type: STS
    dataset:
      type: mteb/sickr-sts
      name: MTEB SICK-R
      config: default
      split: test
    metrics:
    - type: cos_sim_pearson
      value: 78.58086597995997
    - type: cos_sim_spearman
      value: 69.63214182814991
    - type: euclidean_pearson
      value: 72.76175489042691
    - type: euclidean_spearman
      value: 67.84965161872971
    - type: manhattan_pearson
      value: 72.73812689782592
    - type: manhattan_spearman
      value: 67.83610439531277
  - task:
      type: STS
    dataset:
      type: mteb/sts12-sts
      name: MTEB STS12
      config: default
      split: test
    metrics:
    - type: cos_sim_pearson
      value: 75.13970861325006
    - type: cos_sim_spearman
      value: 67.5020551515597
    - type: euclidean_pearson
      value: 66.33415412418276
    - type: euclidean_spearman
      value: 66.82145056673268
    - type: manhattan_pearson
      value: 66.55489484006415
    - type: manhattan_spearman
      value: 66.95147433279057
  - task:
      type: STS
    dataset:
      type: mteb/sts13-sts
      name: MTEB STS13
      config: default
      split: test
    metrics:
    - type: cos_sim_pearson
      value: 78.85850536483447
    - type: cos_sim_spearman
      value: 79.1633350177206
    - type: euclidean_pearson
      value: 72.74090561408477
    - type: euclidean_spearman
      value: 73.57374448302961
    - type: manhattan_pearson
      value: 72.92980654233226
    - type: manhattan_spearman
      value: 73.72777155112588
  - task:
      type: STS
    dataset:
      type: mteb/sts14-sts
      name: MTEB STS14
      config: default
      split: test
    metrics:
    - type: cos_sim_pearson
      value: 79.51125593897028
    - type: cos_sim_spearman
      value: 74.46048326701329
    - type: euclidean_pearson
      value: 70.87726087052985
    - type: euclidean_spearman
      value: 67.7721470654411
    - type: manhattan_pearson
      value: 71.05892792135637
    - type: manhattan_spearman
      value: 67.93472619779037
  - task:
      type: STS
    dataset:
      type: mteb/sts15-sts
      name: MTEB STS15
      config: default
      split: test
    metrics:
    - type: cos_sim_pearson
      value: 83.8299348880489
    - type: cos_sim_spearman
      value: 84.47194637929275
    - type: euclidean_pearson
      value: 78.68768462480418
    - type: euclidean_spearman
      value: 79.80526323901917
    - type: manhattan_pearson
      value: 78.6810718151946
    - type: manhattan_spearman
      value: 79.7820584821254
  - task:
      type: STS
    dataset:
      type: mteb/sts16-sts
      name: MTEB STS16
      config: default
      split: test
    metrics:
    - type: cos_sim_pearson
      value: 79.99206664843005
    - type: cos_sim_spearman
      value: 80.96089203722137
    - type: euclidean_pearson
      value: 71.31216213716365
    - type: euclidean_spearman
      value: 71.45258140049407
    - type: manhattan_pearson
      value: 71.26140340402836
    - type: manhattan_spearman
      value: 71.3896894666943
  - task:
      type: STS
    dataset:
      type: mteb/sts17-crosslingual-sts
      name: MTEB STS17 (en-en)
      config: en-en
      split: test
    metrics:
    - type: cos_sim_pearson
      value: 87.35697089594868
    - type: cos_sim_spearman
      value: 87.78202647220289
    - type: euclidean_pearson
      value: 84.20969668786667
    - type: euclidean_spearman
      value: 83.91876425459982
    - type: manhattan_pearson
      value: 84.24429755612542
    - type: manhattan_spearman
      value: 83.98826315103398
  - task:
      type: STS
    dataset:
      type: mteb/sts22-crosslingual-sts
      name: MTEB STS22 (en)
      config: en
      split: test
    metrics:
    - type: cos_sim_pearson
      value: 69.06962775868384
    - type: cos_sim_spearman
      value: 69.34889515492327
    - type: euclidean_pearson
      value: 69.28108180412313
    - type: euclidean_spearman
      value: 69.6437114853659
    - type: manhattan_pearson
      value: 69.39974983734993
    - type: manhattan_spearman
      value: 69.69057284482079
  - task:
      type: STS
    dataset:
      type: mteb/stsbenchmark-sts
      name: MTEB STSBenchmark
      config: default
      split: test
    metrics:
    - type: cos_sim_pearson
      value: 82.42553734213958
    - type: cos_sim_spearman
      value: 81.38977341532744
    - type: euclidean_pearson
      value: 76.47494587945522
    - type: euclidean_spearman
      value: 75.92794860531089
    - type: manhattan_pearson
      value: 76.4768777169467
    - type: manhattan_spearman
      value: 75.9252673228599
  - task:
      type: Reranking
    dataset:
      type: mteb/scidocs-reranking
      name: MTEB SciDocsRR
      config: default
      split: test
    metrics:
    - type: map
      value: 80.78825425914722
    - type: mrr
      value: 94.60017197762296
  - task:
      type: Retrieval
    dataset:
      type: scifact
      name: MTEB SciFact
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 60.633
    - type: map_at_10
      value: 70.197
    - type: map_at_100
      value: 70.758
    - type: map_at_1000
      value: 70.765
    - type: map_at_3
      value: 67.082
    - type: map_at_5
      value: 69.209
    - type: mrr_at_1
      value: 63.333
    - type: mrr_at_10
      value: 71.17
    - type: mrr_at_100
      value: 71.626
    - type: mrr_at_1000
      value: 71.633
    - type: mrr_at_3
      value: 68.833
    - type: mrr_at_5
      value: 70.6
    - type: ndcg_at_1
      value: 63.333
    - type: ndcg_at_10
      value: 74.697
    - type: ndcg_at_100
      value: 76.986
    - type: ndcg_at_1000
      value: 77.225
    - type: ndcg_at_3
      value: 69.527
    - type: ndcg_at_5
      value: 72.816
    - type: precision_at_1
      value: 63.333
    - type: precision_at_10
      value: 9.9
    - type: precision_at_100
      value: 1.103
    - type: precision_at_1000
      value: 0.11199999999999999
    - type: precision_at_3
      value: 26.889000000000003
    - type: precision_at_5
      value: 18.2
    - type: recall_at_1
      value: 60.633
    - type: recall_at_10
      value: 87.36699999999999
    - type: recall_at_100
      value: 97.333
    - type: recall_at_1000
      value: 99.333
    - type: recall_at_3
      value: 73.656
    - type: recall_at_5
      value: 82.083
  - task:
      type: PairClassification
    dataset:
      type: mteb/sprintduplicatequestions-pairclassification
      name: MTEB SprintDuplicateQuestions
      config: default
      split: test
    metrics:
    - type: cos_sim_accuracy
      value: 99.76633663366337
    - type: cos_sim_ap
      value: 93.84024096781063
    - type: cos_sim_f1
      value: 88.08080808080808
    - type: cos_sim_precision
      value: 88.9795918367347
    - type: cos_sim_recall
      value: 87.2
    - type: dot_accuracy
      value: 99.46336633663367
    - type: dot_ap
      value: 75.78127156965245
    - type: dot_f1
      value: 71.41403865717193
    - type: dot_precision
      value: 72.67080745341616
    - type: dot_recall
      value: 70.19999999999999
    - type: euclidean_accuracy
      value: 99.67524752475248
    - type: euclidean_ap
      value: 88.61274955249769
    - type: euclidean_f1
      value: 82.30852211434735
    - type: euclidean_precision
      value: 89.34426229508196
    - type: euclidean_recall
      value: 76.3
    - type: manhattan_accuracy
      value: 99.67722772277227
    - type: manhattan_ap
      value: 88.77516158012779
    - type: manhattan_f1
      value: 82.36536430834212
    - type: manhattan_precision
      value: 87.24832214765101
    - type: manhattan_recall
      value: 78.0
    - type: max_accuracy
      value: 99.76633663366337
    - type: max_ap
      value: 93.84024096781063
    - type: max_f1
      value: 88.08080808080808
  - task:
      type: Clustering
    dataset:
      type: mteb/stackexchange-clustering
      name: MTEB StackExchangeClustering
      config: default
      split: test
    metrics:
    - type: v_measure
      value: 59.20812266121527
  - task:
      type: Clustering
    dataset:
      type: mteb/stackexchange-clustering-p2p
      name: MTEB StackExchangeClusteringP2P
      config: default
      split: test
    metrics:
    - type: v_measure
      value: 33.954248554638056
  - task:
      type: Reranking
    dataset:
      type: mteb/stackoverflowdupquestions-reranking
      name: MTEB StackOverflowDupQuestions
      config: default
      split: test
    metrics:
    - type: map
      value: 51.52800990025549
    - type: mrr
      value: 52.360394915541974
  - task:
      type: Summarization
    dataset:
      type: mteb/summeval
      name: MTEB SummEval
      config: default
      split: test
    metrics:
    - type: cos_sim_pearson
      value: 24.57438758817976
    - type: cos_sim_spearman
      value: 24.747448399760643
    - type: dot_pearson
      value: 26.589017584184987
    - type: dot_spearman
      value: 25.653620812462783
  - task:
      type: Retrieval
    dataset:
      type: trec-covid
      name: MTEB TRECCOVID
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 0.253
    - type: map_at_10
      value: 2.1399999999999997
    - type: map_at_100
      value: 12.873000000000001
    - type: map_at_1000
      value: 31.002000000000002
    - type: map_at_3
      value: 0.711
    - type: map_at_5
      value: 1.125
    - type: mrr_at_1
      value: 96.0
    - type: mrr_at_10
      value: 98.0
    - type: mrr_at_100
      value: 98.0
    - type: mrr_at_1000
      value: 98.0
    - type: mrr_at_3
      value: 98.0
    - type: mrr_at_5
      value: 98.0
    - type: ndcg_at_1
      value: 94.0
    - type: ndcg_at_10
      value: 84.881
    - type: ndcg_at_100
      value: 64.694
    - type: ndcg_at_1000
      value: 56.85
    - type: ndcg_at_3
      value: 90.061
    - type: ndcg_at_5
      value: 87.155
    - type: precision_at_1
      value: 96.0
    - type: precision_at_10
      value: 88.8
    - type: precision_at_100
      value: 65.7
    - type: precision_at_1000
      value: 25.080000000000002
    - type: precision_at_3
      value: 92.667
    - type: precision_at_5
      value: 90.0
    - type: recall_at_1
      value: 0.253
    - type: recall_at_10
      value: 2.292
    - type: recall_at_100
      value: 15.78
    - type: recall_at_1000
      value: 53.015
    - type: recall_at_3
      value: 0.7270000000000001
    - type: recall_at_5
      value: 1.162
  - task:
      type: Retrieval
    dataset:
      type: webis-touche2020
      name: MTEB Touche2020
      config: default
      split: test
    metrics:
    - type: map_at_1
      value: 2.116
    - type: map_at_10
      value: 9.625
    - type: map_at_100
      value: 15.641
    - type: map_at_1000
      value: 17.127
    - type: map_at_3
      value: 4.316
    - type: map_at_5
      value: 6.208
    - type: mrr_at_1
      value: 32.653
    - type: mrr_at_10
      value: 48.083999999999996
    - type: mrr_at_100
      value: 48.631
    - type: mrr_at_1000
      value: 48.649
    - type: mrr_at_3
      value: 42.857
    - type: mrr_at_5
      value: 46.224
    - type: ndcg_at_1
      value: 29.592000000000002
    - type: ndcg_at_10
      value: 25.430999999999997
    - type: ndcg_at_100
      value: 36.344
    - type: ndcg_at_1000
      value: 47.676
    - type: ndcg_at_3
      value: 26.144000000000002
    - type: ndcg_at_5
      value: 26.304
    - type: precision_at_1
      value: 32.653
    - type: precision_at_10
      value: 24.082
    - type: precision_at_100
      value: 7.714
    - type: precision_at_1000
      value: 1.5310000000000001
    - type: precision_at_3
      value: 26.531
    - type: precision_at_5
      value: 26.939
    - type: recall_at_1
      value: 2.116
    - type: recall_at_10
      value: 16.794
    - type: recall_at_100
      value: 47.452
    - type: recall_at_1000
      value: 82.312
    - type: recall_at_3
      value: 5.306
    - type: recall_at_5
      value: 9.306000000000001
  - task:
      type: Classification
    dataset:
      type: mteb/toxic_conversations_50k
      name: MTEB ToxicConversationsClassification
      config: default
      split: test
    metrics:
    - type: accuracy
      value: 67.709
    - type: ap
      value: 13.541535578501716
    - type: f1
      value: 52.569619919446794
  - task:
      type: Classification
    dataset:
      type: mteb/tweet_sentiment_extraction
      name: MTEB TweetSentimentExtractionClassification
      config: default
      split: test
    metrics:
    - type: accuracy
      value: 56.850594227504246
    - type: f1
      value: 57.233377364910574
  - task:
      type: Clustering
    dataset:
      type: mteb/twentynewsgroups-clustering
      name: MTEB TwentyNewsgroupsClustering
      config: default
      split: test
    metrics:
    - type: v_measure
      value: 39.463722986090474
  - task:
      type: PairClassification
    dataset:
      type: mteb/twittersemeval2015-pairclassification
      name: MTEB TwitterSemEval2015
      config: default
      split: test
    metrics:
    - type: cos_sim_accuracy
      value: 84.09131549144662
    - type: cos_sim_ap
      value: 66.86677647503386
    - type: cos_sim_f1
      value: 62.94631710362049
    - type: cos_sim_precision
      value: 59.73933649289099
    - type: cos_sim_recall
      value: 66.51715039577837
    - type: dot_accuracy
      value: 80.27656911247541
    - type: dot_ap
      value: 54.291720398612085
    - type: dot_f1
      value: 54.77150537634409
    - type: dot_precision
      value: 47.58660957571039
    - type: dot_recall
      value: 64.5118733509235
    - type: euclidean_accuracy
      value: 82.76211480002385
    - type: euclidean_ap
      value: 62.430397690753296
    - type: euclidean_f1
      value: 59.191590539356774
    - type: euclidean_precision
      value: 56.296119971435374
    - type: euclidean_recall
      value: 62.401055408970976
    - type: manhattan_accuracy
      value: 82.7561542588067
    - type: manhattan_ap
      value: 62.41882051995577
    - type: manhattan_f1
      value: 59.32101002778785
    - type: manhattan_precision
      value: 54.71361711611321
    - type: manhattan_recall
      value: 64.77572559366754
    - type: max_accuracy
      value: 84.09131549144662
    - type: max_ap
      value: 66.86677647503386
    - type: max_f1
      value: 62.94631710362049
  - task:
      type: PairClassification
    dataset:
      type: mteb/twitterurlcorpus-pairclassification
      name: MTEB TwitterURLCorpus
      config: default
      split: test
    metrics:
    - type: cos_sim_accuracy
      value: 88.79574649745798
    - type: cos_sim_ap
      value: 85.28960532524223
    - type: cos_sim_f1
      value: 77.98460043358001
    - type: cos_sim_precision
      value: 75.78090948714224
    - type: cos_sim_recall
      value: 80.32029565753002
    - type: dot_accuracy
      value: 85.5939767920208
    - type: dot_ap
      value: 76.14131706694056
    - type: dot_f1
      value: 72.70246298696868
    - type: dot_precision
      value: 65.27012127894156
    - type: dot_recall
      value: 82.04496458269172
    - type: euclidean_accuracy
      value: 86.72332828812046
    - type: euclidean_ap
      value: 80.84854809178995
    - type: euclidean_f1
      value: 72.47657499809551
    - type: euclidean_precision
      value: 71.71717171717171
    - type: euclidean_recall
      value: 73.25223283030489
    - type: manhattan_accuracy
      value: 86.7563162184189
    - type: manhattan_ap
      value: 80.87598895575626
    - type: manhattan_f1
      value: 72.54617892068092
    - type: manhattan_precision
      value: 68.49268225960881
    - type: manhattan_recall
      value: 77.10963966738528
    - type: max_accuracy
      value: 88.79574649745798
    - type: max_ap
      value: 85.28960532524223
    - type: max_f1
      value: 77.98460043358001
---