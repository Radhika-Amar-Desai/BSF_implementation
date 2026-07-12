# Grassmann BSF

## Mathematical Pre-requisites

### Basis and Orthonormal Basis

A **basis** of a vector space is a set of **linearly independent vectors** that span the entire space. A basis **does not need to be orthogonal**.

For example, the following vectors form a valid basis for $\mathbb{R}^2$:

$
\begin{bmatrix}
1\\0
\end{bmatrix},
\qquad
\begin{bmatrix}
1\\1
\end{bmatrix}
$

since they are linearly independent and span the plane.

An **orthonormal basis** is a special type of basis in which:

- Every basis vector has unit norm.
- Every pair of basis vectors is orthogonal.

For example,

$
\begin{bmatrix}
1\\0
\end{bmatrix},
\qquad
\begin{bmatrix}
0\\1
\end{bmatrix}
$

is an orthonormal basis of $\mathbb{R}^2$.

---

### Stiefel Manifold

A **Stiefel manifold**, denoted by $St(b,d)$ is the set of all possible **orthonormal $b$-frames** in $\mathbb{R}^d$.

Equivalently, each point on the Stiefel manifold is a matrix $D \in \mathbb{R}^{b \times d}$ whose rows (or columns, depending on convention) are orthonormal: $DD^\top = I_b.$

Thus, every point on the Stiefel manifold represents **one particular orthonormal basis** of a $b$-dimensional subspace.

For example,

$
D=
\begin{bmatrix}
1 & 0 & 0\\
0 & 1 & 0
\end{bmatrix}
$

and

$
D'=
\begin{bmatrix}
\frac1{\sqrt2} & \frac1{\sqrt2} & 0\\
-\frac1{\sqrt2} & \frac1{\sqrt2} & 0
\end{bmatrix}
$

are two different points on $St(2,3)$.

---

### Grassmann Manifold

The **Grassmann manifold**, denoted by $Gr(b,d),$

is the set of all $b$-dimensional **subspaces** of $\mathbb{R}^d$.

Unlike the Stiefel manifold, the Grassmann manifold **does not care which basis is used to represent a subspace**.

For example, both matrices

$$
D=
\begin{bmatrix}
1 & 0 & 0\\
0 & 1 & 0
\end{bmatrix}
$$

and

$$
D'=
\begin{bmatrix}
\frac1{\sqrt2} & \frac1{\sqrt2} & 0\\
-\frac1{\sqrt2} & \frac1{\sqrt2} & 0
\end{bmatrix}
$$

span the same $xy$-plane.

Therefore,

- they are **different points** on the **Stiefel manifold**,
- but they correspond to the **same point** on the **Grassmann manifold**.

---

### Relationship Between Stiefel and Grassmann Manifolds

A single subspace can be represented using infinitely many orthonormal bases obtained by rotating the basis vectors within the subspace.

If $Q$ is any $b \times b$ orthogonal matrix,

$
D' = QD
$

represents a different orthonormal basis, but

$
\operatorname{span}(D') = \operatorname{span}(D).
$

Therefore, many different points on the Stiefel manifold correspond to one point on the Grassmann manifold.

Mathematically,

$
Gr(b,d)=St(b,d)/O(b),
$

where $O(b)$ is the set of all orthogonal $b \times b$ matrices.

This is called a **quotient relationship**.

Intuitively, the quotient operation means:

> Treat all orthonormal bases related by an orthogonal rotation as representing the same object, since they span the same subspace.

---

### Connection to Grassmann BSF

The theoretical motivation of the paper is that **each concept manifold lies within a low-dimensional subspace**.

The decoder block $D_g \in \mathbb{R}^{b \times d}$ therefore represents an orthonormal basis of such a subspace.

The Grassmann BSF constrains every decoder block to satisfy $D_gD_g^\top = I,$ ensuring that each block remains an orthonormal basis throughout training.

Although the optimization is performed over **Stiefel manifolds** (orthonormal bases), the actual object of interest is the **subspace itself**, since different orthonormal bases spanning the same subspace represent the same concept.

This is why the method is called **Grassmannian BSF** even though the decoder is parameterized using points on the **Stiefel manifold**.

## Understanding the Block-wise Operations

After the encoder, we obtain the latent representation: $z = xW + b,$

which is partitioned into blocks. Each block $z_g \in \mathbb{R}^b$ represents **one concept**.

---

### 1. Block Projection ($\Pi_k$)

Instead of selecting the top-$k$ **neurons** (as in TopK SAE), BSF selects the top-$k$ **blocks (concepts)**.

The activation strength of a concept is measured using its **L2 norm**: $\|z_g\|_2.$

Only the $k$ blocks with the largest norms are kept, while the remaining blocks are set to zero.

Thus, $\|z\|_{2,0}$

simply counts the **number of active blocks** (blocks with non-zero L2 norm).

---

### 2. Block Soft Threshold (Group Lasso BSF)

Instead of selecting exactly $k$ concepts, Group Lasso softly suppresses weak concepts using

$$
\operatorname{sh}_\theta(z_g)=
\max\left(1-\frac{\theta}{\|z_g\|_2},0\right)z_g.
$$

**Intuition**

- If the block norm is **small**, the entire block becomes zero.
- If the block norm is **large**, the whole block is scaled down uniformly.

Unlike ReLU, every coordinate in the block is multiplied by the **same scaling factor**.

Example:

```text
Original block
(3, 4, 1)

↓

Shrink by 0.8

↓

(2.4, 3.2, 0.8)
```

The **relative coordinates remain unchanged**. This means the point moves **towards the origin** while staying at the **same location within the concept manifold**. Only the **strength of the concept** decreases.

---

### 3. Why is only the block norm used?

Both Top-$k$ projection and block soft-thresholding depend **only on the block's L2 norm**.

This makes the operations **rotation invariant**.

If the basis vectors inside a subspace are rotated,

- the individual coordinates change,
- but the block norm remains unchanged.

Therefore, the model selects **subspaces (concepts)** rather than a particular choice of basis vectors.

---

### 4. Why not use ReLU or JumpReLU?

ReLU acts **coordinate-wise**.

It forces negative coordinates to become zero, restricting the latent code to only **non-negative combinations** of the basis vectors.

Geometrically:

- **BSF:** allows movement anywhere inside the learned subspace.
- **ReLU:** restricts the representation to only one cone (positive orthant) of that subspace.

Since the paper assumes that concepts occupy an entire low-dimensional subspace (not just a cone), coordinate-wise nonlinearities are incompatible with this assumption.

## Core Idea

Unlike Vanilla BSF, Grassmannian BSF constrains every decoder block to be an **orthonormal basis** of a low-dimensional concept subspace.

Mathematically,

$$
D_gD_g^\top = I.
$$

This means each decoder block is no longer an arbitrary basis but an **orthonormal frame** representing a point on the **Stiefel manifold**, which in turn represents a subspace (Grassmann manifold).

---

## Why can the encoder and decoder be tied?

The decoder defines the basis vectors of a concept subspace.

Suppose a decoder block is

$$
D_g=
\begin{bmatrix}
d_1\\
d_2\\
\vdots\\
d_b
\end{bmatrix}.
$$

Given an input activation \(x\), the encoder's job is to compute the coordinates of \(x\) within this basis.

For an **orthonormal basis**, these coordinates are obtained simply by taking dot products with each basis vector:

$$
z_g = xD_g^\top.
$$

This follows directly from linear algebra.

If

$$
x=\sum_i c_i d_i,
$$

then

$$
x\cdot d_j
=
\sum_i c_i(d_i\cdot d_j)
=
c_j,
$$

because

$$
d_i\cdot d_j=
\begin{cases}
1 & i=j\\
0 & i\neq j.
\end{cases}
$$

Thus, for an orthonormal basis,

$$
\boxed{\text{Projection onto the basis} = \text{Coordinates in that basis}.}
$$

Hence, there is no need to learn an independent encoder matrix—the encoder can simply be the transpose of the decoder.

---

## Why is this not possible in Vanilla BSF?

Vanilla BSF allows decoder blocks to be arbitrary basis vectors.

Suppose

$$
x = 3d_1 + 5d_2,
$$

where \(d_1\) and \(d_2\) are **not orthogonal**.

Then,

$$
x\cdot d_1
=
3(d_1\cdot d_1)
+
5(d_2\cdot d_1),
$$

where the second term is generally non-zero.

Therefore,

$$
xD^\top
$$

does **not** recover the true coordinates, since different basis vectors interfere with one another.

As a result, Vanilla BSF requires a separate learnable encoder \(W\) to estimate the latent coordinates.

---

## Reconstruction

Once the coordinates are computed,

$$
z_g = xD_g^\top,
$$

the contribution of that concept is reconstructed as

$$
m_g = z_gD_g.
$$

Geometrically, this corresponds to:

```text
Input activation
        │
        ▼
Project onto learned orthonormal basis
(compute coordinates)
        │
        ▼
Latent coordinates
        │
        ▼
Reconstruct using the same basis
        │
        ▼
Contribution in the original DINO space
```

Thus, Grassmannian BSF treats the encoder as the **coordinate computation step** and the decoder as the **basis defining the concept subspace**, making the entire architecture closely aligned with the paper's geometric interpretation of concepts as low-dimensional subspaces.