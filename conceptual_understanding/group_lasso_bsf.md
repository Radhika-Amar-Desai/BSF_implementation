# Group Lasso BSF

## Motivation

Can we have a BSF model decide for itself how many number of blocks are to be selected ? We can formulate this behaviour of selection by L2,1 norm which refers to applying L1 norm of L2 norm of the blocks.

$
Loss = Reconstruction Loss + \lambda * \Sigma norm_{L2}(blocks)
$

We use proximal operator here. We try to minimize a function that has 2 parts: a smooth, easy-to-differentiate (like tracking raw data) and a rugged, non-differentiable part(like penalty function for sparsity).

$
prox_f(u) = arg min_u(1/2 * (||u - v||^2_2) + f(u))
$

### What is subdifferential ?

Sometimes, parts of function are rugged and non-differential. Such parts like sharp corners hv a variety of valid slopes and you can fit an entire family of tangent lines underneath the corner that touch it without crossing the function. This collection of valid slopes is called the subdifferential.

$
if u_g \neq 0:
$
$
subdiff(||u_g||_2) = \frac{u_g}{||u_g||_2}
$

$
if u_g = 0:
$
$
subdiff(||u_g||2) = {z: ||z||_2 \leq 1}
$

## Derivation of the Block Soft-Thresholding Operator via Subdifferentials

To find the closed-form expression for the proximal operator of the $L_2$ norm penalty, we solve the optimization problem block by block. For a given block $g$, the problem is formulated as:

$$
\hat{u}_g = \arg\min_{u_g} \left\{ \frac{1}{2} \|u_g - v_g\|_2^2 + \lambda \|u_g\|_2 \right\}
$$

---

### 1. The Optimality Condition

Let $F(u_g) = \frac{1}{2} \|u_g - v_g\|_2^2 + \lambda \|u_g\|_2$. Because $F(u_g)$ is a convex function, a point $\hat{u}_g$ is a global minimizer if and only if the zero vector belongs to its subdifferential:

$$
0 \in \partial F(\hat{u}_g)
$$

Applying the sum rule for subdifferentials allows us to separate the differentiable loss function from the rugged penalty function:

$$
0 \in (\hat{u}_g - v_g) + \lambda \partial \|\hat{u}_g\|_2
$$

Rearranging the terms yields our fundamental equilibrium equation:

$$
v_g - \hat{u}_g \in \lambda \partial \|\hat{u}_g\|_2
$$

---

### 2. Case Analysis Using Subdifferentials

Recall the definition of the subdifferential for the $L_2$ norm:

$$
\partial \|u_g\|_2 = \begin{cases} \left\{ \frac{u_g}{\|u_g\|_2} \right\} & \text{if } u_g \neq 0 \\ \{z : \|z\|_2 \leq 1\} & \text{if } u_g = 0 \end{cases}
$$

#### **Case 1: The Thresholding Condition ($\hat{u}_g = 0$)**

If the optimal vector is shrunk completely to zero ($\hat{u}_g = 0$), we substitute this into the equilibrium equation:

$$
v_g - 0 \in \lambda \partial \|0\|_2
$$

$$
v_g \in \lambda \{z : \|z\|_2 \le 1\}
$$

$$
v_g \in \{z' : \|z'\|_2 \le \lambda\}
$$

Taking the magnitude ($L_2$ norm) of both sides provides the exact condition for elimination:

$$
\|v_g\|_2 \le \lambda
$$

> **Conclusion 1:** If the magnitude of the input vector $v_g$ does not exceed the penalty threshold $\lambda$, it falls into the sharp corner of the prior and is set entirely to zero:
>
> $$
> \hat{u}_g = 0 \quad \text{for} \quad \|v_g\|_2 \le \lambda
> $$

#### **Case 2: The Shrinkage Condition ($\hat{u}_g \neq 0$)**

If the vector survives the threshold ($\hat{u}_g \neq 0$), the subdifferential contains a single valid slope:

$$
v_g - \hat{u}_g = \lambda \frac{\hat{u}_g}{\|\hat{u}_g\|_2}
$$

We isolate $v_g$ by factoring out $\hat{u}_g$:

$$
v_g = \hat{u}_g \left(1 + \frac{\lambda}{\|\hat{u}_g\|_2}\right)
$$

This structural relationship tells us that $v_g$ and $\hat{u}_g$ point in the exact same direction (they are collinear). To determine the scaling factor, we take the $L_2$ norm of both sides:

$$
\|v_g\|_2 = \left\| \hat{u}_g \left(1 + \frac{\lambda}{\|\hat{u}_g\|_2}\right) \right\|_2
$$

Because $\left(1 + \frac{\lambda}{\|\hat{u}_g\|_2}\right)$ is a positive scalar value, it can be pulled straight out of the norm:

$$
\|v_g\|_2 = \|\hat{u}_g\|_2 \left(1 + \frac{\lambda}{\|\hat{u}_g\|_2}\right)
$$

$$
\|v_g\|_2 = \|\hat{u}_g\|_2 + \lambda
$$

Solving for the magnitude of our optimal vector yields:

$$
\left\|\hat{u}_g\right\|_2 = \|v_g\|_2 - \lambda
$$

Since we began with the assumption that $\hat{u}_g \neq 0$, this confirms the condition holds strictly when $\|v_g\|_2 > \lambda$.

To map this magnitude back to the original directional vector, we substitute $\|\hat{u}_g\|_2$ back into our collinearity relation:

$$
v_g = \hat{u}_g \left(1 + \frac{\lambda}{\|v_g\|_2 - \lambda}\right)
$$

$$
v_g = \hat{u}_g \left(\frac{\|v_g\|_2 - \lambda + \lambda}{\|v_g\|_2 - \lambda}\right)
$$

$$
v_g = \hat{u}_g \left(\frac{\|v_g\|_2}{\|v_g\|_2 - \lambda}\right)
$$

Finally, resolving for $\hat{u}_g$ gives us:

$$
\hat{u}_g = v_g \left(\frac{\|v_g\|_2 - \lambda}{\|v_g\|_2}\right) = v_g \left(1 - \frac{\lambda}{\|v_g\|_2}\right)
$$

> **Conclusion 2:** If the input vector magnitude is greater than $\lambda$, its direction is preserved while its length is scaled down linearly by exactly $\lambda$:
>
> $$
> \hat{u}_g = v_g \left(1 - \frac{\lambda}{\|v_g\|_2}\right) \quad \text{for} \quad \|v_g\|_2 > \lambda
> $$

---

### 3. The Unified Block Soft-Thresholding Operator

By utilizing the positive-part operator $(x)_+ = \max(0, x)$, we can condense both cases into a single, unified mathematical function known as the **Block Soft-Thresholding** (or Shrinkage) operator:

$$
\hat{u}_g = \text{prox}_{\lambda \|\cdot\|_2}(v_g) = \left(1 - \frac{\lambda}{\|v_g\|_2}\right)_+ v_g
$$

---

### 4. Changes for Practical Implementation

The authors developed two versions of the Group Lasso BSF to address the gap between theoretical derivation and practical training stability. While the **"Paper version"** ($paper\_version=True$) implements the exact **convex relaxation** of the MAP objective, the **"Default version"** ($paper\_version=False$) introduces several engineering modifications to ensure the model trains reliably.

#### 1. Transition to Block JumpReLU
The primary change for practical implementation is replacing the continuous soft-thresholding activation with a **block JumpReLU**. 
*   **Theoretical Version:** Uses the **soft-threshold function ($sh_\theta$)**, which is the proximal operator of the $\ell_{2,1}$ norm and induces sparsity through **shrinkage**.
*   **Practical Version:** Uses a **hard block gate $H(\|a_g\| - \theta)$**. A block "fires" only when its $\ell_2$ norm exceeds a specific per-block threshold ($\theta$).

#### 2. Elimination of Magnitude Shrinkage
In the practical version, the authors found that the **magnitude shrinkage** required by the theoretical $\ell_{2,1}$ penalty **collapses under a norm-constrained decoder**. 
*   **The Change:** Once a block clears the threshold $\theta$, the **full signed code is kept** rather than being reduced by $\theta$.
*   **Rationale:** This ensures the featurizer provides a strong enough signal for the decoder to reconstruct the original activation without "starving" the reconstruction quality.

#### 3. Learning Thresholds via Straight-Through Estimator (STE)
Because the hard JumpReLU gate is non-differentiable (its derivative is zero almost everywhere), the authors implemented a **Straight-Through Estimator** to learn the optimal threshold for each block.
*   **Rectangle-Kernel Pseudo-Derivative:** This "fake" gradient carries the signal specifically to the **threshold $\theta$**, allowing the model to learn when a concept should be active.
*   **Gradient Path Separation:** The implementation passes **no gradient to the raw magnitude $\|a_g\|$** through the gate; instead, the encoder learns to produce correct coordinates through the separate "magnitude path" ($z = a \times gate$).

#### 4. Penalty Shift and Initialization
The practical implementation shifts the focus from smooth coordinate penalties to discrete block counts:
*   **$L_0$ Penalty:** Instead of a Group Lasso ($\ell_{2,1}$) penalty on the sum of magnitudes, the default version uses an **$L_0$ (active-block) penalty** to control the overall sparsity level.
*   **Adaptive Initialization:** The thresholds ($\theta$) are **initialized from the first training batch** so that approximately the `target_l0` number of blocks are active from the start of training.

#### 5. Preservation of Signed Codes
Across both versions, the authors removed the standard ReLU activation in the encoder. They use a **free linear encoder that produces signed per-block codes**. This allows the featurizer to resolve the full geometry of a manifold in its designated subspace rather than being restricted to a positive-only "cone".