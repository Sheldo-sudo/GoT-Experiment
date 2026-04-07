---
license: apache-2.0
task_categories:
- text-generation
- graph-ml
language:
- en
size_categories:
- 1K<n<10K
configs:
- config_name: cycle
  data_files:
  - split: test
    path: cycle_test.json
- config_name: connectivity
  data_files:
  - split: test
    path: connectivity_test.json
- config_name: flow
  data_files:
  - split: test
    path: flow_test.json
- config_name: bipartite
  data_files:
  - split: test
    path: bipartite_test.json
- config_name: hamilton
  data_files:
  - split: test
    path: hamilton_test.json
- config_name: shortest
  data_files:
  - split: test
    path: shortest_test.json
- config_name: topology
  data_files:
  - split: test
    path: topology_test.json
- config_name: substructure
  data_files:
  - split: test
    path: substructure_test.json
- config_name: triangle
  data_files:
  - split: test
    path: triangle_test.json
---


