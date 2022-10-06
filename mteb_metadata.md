---
tags:
- mteb
model-index:
- name: sgpt-bloom-1b3-nli
  results:
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_reviews_multi
      name: MTEB AmazonReviewsClassification (fr)
      config: fr
      split: test
      revision: c379a6705fec24a2493fa68e011692605f44e119
    metrics:
    - type: accuracy
      value: 39.286
    - type: f1
      value: 38.87078070073539
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_reviews_multi
      name: MTEB AmazonReviewsClassification (zh)
      config: zh
      split: test
      revision: c379a6705fec24a2493fa68e011692605f44e119
    metrics:
    - type: accuracy
      value: 37.634
    - type: f1
      value: 36.86046604093418
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_domain
      name: MTEB MTOPDomainClassification (fr)
      config: fr
      split: test
      revision: a7e2a951126a26fc8c6a69f835f33a346ba259e3
    metrics:
    - type: accuracy
      value: 83.79893517068588
    - type: f1
      value: 83.72326662566203
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_intent
      name: MTEB MTOPIntentClassification (fr)
      config: fr
      split: test
      revision: 6299947a7777084cc2d4b64235bf7190381ce755
    metrics:
    - type: accuracy
      value: 63.36047604134043
    - type: f1
      value: 44.261707019308126
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (fr)
      config: fr
      split: test
      revision: 072a486a144adf7f4479a4a0dddb2152e161e1ea
    metrics:
    - type: accuracy
      value: 64.57632817753867
    - type: f1
      value: 62.60453982786661
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (fr)
      config: fr
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 69.59986550100874
    - type: f1
      value: 69.71803697939914
  - task:
      type: STS
    dataset:
      type: mteb/sts22-crosslingual-sts
      name: MTEB STS22 (zh)
      config: zh
      split: test
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
    metrics:
    - type: cos_sim_pearson
      value: 59.71781185663265
    - type: cos_sim_spearman
      value: 58.538648447630514
    - type: euclidean_pearson
      value: 53.53848180206165
    - type: euclidean_spearman
      value: 56.33730262964236
    - type: manhattan_pearson
      value: 54.62109820575505
    - type: manhattan_spearman
      value: 57.223846291318914
  - task:
      type: STS
    dataset:
      type: mteb/sts22-crosslingual-sts
      name: MTEB STS22 (fr)
      config: fr
      split: test
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
    metrics:
    - type: cos_sim_pearson
      value: 73.44021434651606
    - type: cos_sim_spearman
      value: 73.13412769502769
    - type: euclidean_pearson
      value: 68.16368597409867
    - type: euclidean_spearman
      value: 72.44964781564485
    - type: manhattan_pearson
      value: 69.42307032478939
    - type: manhattan_spearman
      value: 73.3523195012387
---