# Single Concept F1-score pipeline

We take ImageNet1-k dataset and obtain patch-wise Dino-v3 activations. The pipeline for evaluation is as follows:

* For each class, we find the best block, which is the one that yields maximum F1 score.

* We find F1 score for a particular block by the following process.

* For one block, take all firing patches and train a logistic regression for the class c. Now, we obtain $w^Tx + b  = 0$ as the decision boundary and vector $w$.

* For all images we take maximum value for the score as projection of the co-ordinates $z_g$ for a block g to the vector $w$ for their patches.

* After we obtain image scores, we adjust the confidence threshold to the one that yields maximum F1 score.

## Concept map smoothness

* For each image, pass through BSF and for 8 highest firing blocks, take patch activations to an concept-activation map, which is unit-l2 normalized.

* We calculate total variation as $\Sigma_{i} \Sigma_{j \in neighbours}|A_i-A_j|$ for each block and divide the sum of blocks by 8.

* We calculate dirchlet energy as $\Sigma_{i} \Sigma_{j \in neighbours}(A_i-A_j)^2$ for each block and divide the sum of blocks by 8.

### Why do we need both TV and Dirichlet energy ? How do they differ ?

* TV doesn't differentiate if change was added up or were there many small bumps. For example, a change of 0.1 10 times or a change of 1, will be the same quantity in TV.

* Dirichlet likes gradual, small changes. A change of 0.1 10 times has a DE of 0.1 while a change of 1 has a DE of 1.

* DE is particularly harsh towards sudden spikes and favours gradual transitions throughout the activation map.