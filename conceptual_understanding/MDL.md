# MDL

## Motivation

Best representation of data comes is the one that uses least number of bits to describe the data.

## Block Support Featurizer

We need to communicate activations. We have activation vector $x \approx Dz$.

Now, we need to communicate the following:

- What blocks are active ?
- Co-ordinates inside those blocks. $(z_g)$

### What needs to be transmitted ?

- What blocks are active ? (Support term)
- Co-ordinate values inside those blocks (Code term)
- Reconstruction error (Residual term)

## MDL Formula

$L(x) = L(support) + L(coefficient\_cost) + L(resdiual)$

### Support Term

Encodes which blocks are active. We use binomial formula. This problem is equivalent to choosing $k$ objects from a collection of $m$.

$Support\_term = log_2(C^m_k)$

### Code Term

Encodes the exact co-ordinates for active blocks.
$Code\_Term = \Sigma p_g \Sigma \frac{1}{2} (log_2 ( 1 + \frac{\sigma_j^2}{distortion}))_+$

Here, $p_g$ = probability of block getting fired
$\sigma$ = eigen value of covariance matrix of the block

We assume that the code is a gaussian variable and according to shannon distortion theory the formula is $1 + \frac{1}{2}{\frac{\sigma_j^2}{distortion}}$ and the function $()_+$ is $max(., 0)$ to avoid negative bits we simply ignore the components whose variance is smaller than distortion.

**Why smooth gaussian instead of gaussian ?**

* We don't want discounties, in $log_2(\frac{\sigma_j^2}{distortion})$ if the variance < distortion than we get negative numbers this problem is mitigated by adding 1 to the term $log_2(\frac{\sigma_j^2}{distortion})$ or by clipping the value to zero.
* Later on if MDL is to be used for a gradient descent based search method to find the most appropriate block size $b$, and dictionary size $D$, we would prefer an easily differntiable function instead of having blind spots.
* Metric shouldn't cut off at a threshold but behave smoothly everywhere.

### Residual Term

Encodes the reconstruction error.

$Residual\_term = \Sigma \frac{1}{2} log_2 \frac{\lambda_a^2}{distortion}$

