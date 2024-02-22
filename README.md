# QTAIM-Embed
<img src="https://github.com/santi921/qtaim_embed/blob/main/data/plots/TOC.png" width=80% height=80%>
A GNN package for molecular properties. These models can handle spin and charged species as well as complex atom, bond features. 
QTAIM features are compatible via a sister package <a href="https://github.com/santi921/qtaim_generator">Generator</a> and can add performance and robustness to existing models 
Currently only structure-to-property models are supported but we are working on structure-to-node level models. 
<br/>

The current implementation supports a good starting set of message-passing functions:
- GCN
- GAT
- GCN+Residual
  
In addition, several global readout functions are implemented for size-intensive and extensive properties including:
- Mean
- Sum
- WeightedMean
- WeightedSum
- GlobalAttentionPooling
- Set2Set  


TODO: some instructional notebooks 
