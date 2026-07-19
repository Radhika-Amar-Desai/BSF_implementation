# Mathematical Foundation: Block Sparsity & Concept Manifolds

## 1. The Big Picture
The core hypothesis is that neural activations ($x$) are a **sparse sum of points** drawn from a few active concept **manifolds** ($M_i$). 
*   **The Problem:** Traditional Sparse Autoencoders (SAEs) treat concepts as single directions. If a concept is actually a multi-dimensional manifold (like a "rabbit face"), a single direction cannot capture its internal geometry.
*   **The Solution:** Use **Block Sparsity**. This assumes that concepts live in low-dimensional **subspaces** ($V_g$), allowing us to recover both the concept's presence and its internal coordinates.

---

## 2. The Generative Model
We model an activation $x$ as:
$$x = \sum_{g \in S} z_g D_g + \epsilon$$
*   **$z_g$**: The coordinate vector for block $g$.
*   **$D_g$**: The "dictionary" or frame for that concept's subspace.
*   **$\epsilon$**: Gaussian noise, $\epsilon \sim \mathcal{N}(0, \sigma^2 I_d)$.
*   **$S$**: The set of active blocks (the "support").

---

## 3. Deriving the Likelihood (Reconstruction Error)
Why does the model use **Mean Squared Error (MSE)**? It follows directly from the assumption that the noise ($\epsilon$) is **Gaussian**.

1.  **Probability of Noise:** The probability density of Gaussian noise is $p(\epsilon) \propto \exp\left(-\frac{\|\epsilon\|_2^2}{2\sigma^2}\right)$.
2.  **Substitution:** Since $\epsilon = x - zD$, the likelihood of seeing activation $x$ given coordinates $z$ is:
    $$p(x|z) \propto \exp\left(-\frac{\|x - zD\|_2^2}{2\sigma^2}\right)$$
3.  **Log-Likelihood:** Taking the natural log turns the exponent into a linear term:
    $$\log p(x|z) = -\frac{1}{2\sigma^2} \|x - zD\|_2^2 + \text{const}$$
This shows that maximizing the likelihood is mathematically the same as **minimizing the squared reconstruction error**.

---

## 4. Deriving the Prior (The "Spike-and-Slab")
The **Prior** represents our assumption about $z$ before seeing the data: most concepts are "off," but some are "on".

### The Distribution: $p(z_g) = (1-\pi) \delta_0 + \pi \rho$
*   **$\pi$ (Pi):** This is **not** 3.14. It is a "mixing weight" representing the probability that a concept is active.
*   **The Spike ($(1-\pi) \delta_0$):** With probability $(1-\pi)$, the block is exactly zero (inactive).
*   **The Slab ($\pi \rho$):** With probability $\pi$, the block is active. Its coordinates are distributed uniformly over a **ball** ($B_R$) of radius $R$ centered at the origin.
    *   **Density ($\rho$):** To make the total probability sum to 1, the density must be $\rho = 1/\text{vol}(B_R)$.

### Log-Prior for the Dictionary
Because we assume blocks are independent, the total probability is the **product** of all $G$ individual block probabilities. Taking the log turns this into a **sum**:
$$\log p(z) = \sum_{g=1}^G \log p(z_g)$$

If $K$ blocks are active ($z_g \neq 0$) and $(G-K)$ are inactive ($z_g = 0$):
$$\log p(z) = \underbrace{(G - K) \log(1 - \pi)}_{\text{Inactive costs}} + \underbrace{K (\log \pi - \log \text{vol}(B_R))}_{\text{Active costs}}$$

Simplifying this reveals the block-sparsity penalty $\|z\|_{2,0}$ (the count of active blocks):
$$\log p(z) = G \log(1 - \pi) + \|z\|_{2,0} \log \left( \frac{\pi}{(1 - \pi) \text{vol}(B_R)} \right)$$

---

## 5. Final MAP Estimate & The Penalty ($\lambda$)
To find the best code $\hat{z}$, we maximize the posterior probability, which is equivalent to minimizing the negative log-likelihood plus the negative log-prior.

### MAP estimate
This refers to Maximum A Posterior estimate. Here, we try to estimate what value of latent variable would maximize the likelihood of the observed data along with the probability of that latent variable occuring. 

$p(z|x) = \frac{p(x|z) * p(z)}{p(x)}$ implies $p(z|x) = k * p(x|z)p(z)$

We try minimizing $-log(p(z|x))$.

Thus, $$argmin_z (-log(p(z|x)))$$ is our objective function.
$$-log(p(z|x)) = -log(p(x|z)) -log(p(z))$$

Here, $log(p(x|z))$ refers to likelihood and $log(p(z))$ is the prior

### The Objective Function:
$$\hat{z} = \text{argmin}_z \frac{1}{2} \|x - zD\|_2^2 + \lambda \|z\|_{2,0}$$

### The Meaning of $\lambda$:
The regularization parameter $\lambda$ is the "fixed cost" of switching a factor on.
$$\lambda = \sigma^2 \log \left( \frac{(1 - \pi) \text{vol}(B_R)}{\pi} \right)$$
*   If the reconstruction error doesn't drop by at least $\lambda$, the model keeps the block at zero.
*   **Volume scaling:** A concept is only granted $b$ dimensions if it "earns" them by improving reconstruction enough to pay for the volume of that coordinate space.

---

## 6. Key Interpretation Doubts Clarified
*   **Why $\ell_2$ norm?** A manifold has no "preferred" direction. If you rotate the coordinate system, the concept is the same. The only rotation-invariant way to measure the "size" of a block is its $\ell_2$ norm ($\|z_g\|_2$).
*   **Probabilities vs. Densities:** $\pi$ and $(1-\pi)$ are discrete probabilities (the switch), while $\rho$ is a continuous density (the location). We multiply them because the probability of being at a specific "on" coordinate is (Prob of being on) $\times$ (Density at that point).
*   **Evaluation vs. Integration:** We use integrals to define the density $\rho$ so it sums to 1. But during optimization (MAP), we just evaluate the "height" of the probability at a single point to find the most likely answer.

---

## 7. NP-hardness of MAP objective function
* **Combinatorial Search** : The penalty term $||z_g||_{2,0}$ counts the number of active blocks and selecing $k$ blocks from $G$ blocks is a combinatorial search problem.
* **No closed form solution**: For selection of objects from a collection, we don't have any closed form solutions.
* **Continous vs Discrete**: Predicting values of co-ordinates is optimizing a smooth, continous function but selecting the blocks which are to be activated is a discrete problem

## 8. Approximations for solving MAP objective function
* **Relaxing the penalty (*Group Lasoo BSF*)**: The penalty term used here is $l_{2,1}$ norm instead of hard $||z_g||_{2,0}$ which is a convex surrogate (turning a difficult, non-convex penalty into a mathematically easy alternative). The trade-off we make is that the model can try and shrink the co-ordinates of inactive blocks instead of shutting them off at a hard threshold.

* **Greedy Selection (Vanilla and Grassmannian BSFs)**: Here block projection operation is used to select the best k blocks based on their block o