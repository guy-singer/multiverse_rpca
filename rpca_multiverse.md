# Robust PCA (RPCA) for Enigma's Multiverse Model

---

## 1  RPCA

### 1.1  From Ordinary PCA to Robust Decomposition
* **Standard PCA** factorises a centred data matrix  
  $D\in\mathbb{R}^{m\times n}$ as $D = U\Sigma V^{\!\top}$ and retains the top-$r$ singular vectors.  
* One gross outlier can rotate every principal component because the least-squares objective  
  $\displaystyle \min_{\operatorname{rank}(L)\le r}\lVert D-L\rVert_F^2$  
  assumes small, Gaussian noise.

### 1.2  The RPCA Model  
Robust PCA assumes the **additive, two-part** model  

$
\boxed{D \;=\; L \;+\; S},
$

| Symbol | Meaning | Constraint |
|--------|---------|------------|
| $L$    | Global, coherent structure | **Low rank** |
| $S$    | Rare, possibly large deviations | **Sparse** (most entries $=0$) |

### 1.3  Principal-Component Pursuit (PCP)

RPCA is cast as the convex programme  

$
\min_{L,S}\; \lVert L\rVert_* \;+\; \lambda\,\lVert S\rVert_1
\quad \text{s.t.} \quad
D \;=\; L \;+\; S,
$

where  

* $\lVert L\rVert_* \;=\; \sum_i \sigma_i(L)$ (the **nuclear norm**) promotes low rank,  
* $\lVert S\rVert_1$ promotes sparsity,  
* $\lambda \approx 1/\sqrt{\max(m,n)}$ balances the two terms.

**Exact-recovery theorem** (Candès et al., 2011): if  

1. $L$ is *incoherent* with the coordinate axes, and  
2. $S$ has at most $\rho mn$ non-zeros (small $\rho$),

then PCP returns the true $L$ and $S$.

### 1.4  Algorithm Families (Sketch)

| Family | Core idea | Per-iteration cost |
|--------|-----------|--------------------|
| Augmented-Lagrange / Inexact ALM | Alternate **soft-threshold $S$** and **SV-shrink $L$** | $O(mn\min\{m,n\})$ |
| ADMM | Split (1), parallelise updates | Similar to ALM |
| Non-convex factorisation | Write $L = AB^{\!\top}$ with $r\ll\min(m,n)$; SGD on $(A,B)$ | $O((m+n)r^2)$ |
| Streaming (GRASTA, incPCP) | Incrementally update a low-dim subspace | $O(dr)$ per sample |

---

## 2  Why RPCA Helps Enigma’s **Multiverse** World Model

| Multiverse pain-point | RPCA viewpoint | Mitigation |
|-----------------------|----------------|------------|
| **Cross-view consistency** (two cameras should see the same crash) | Stack the two $3$-channel images → $6$-channel column; shared 3-D scene is **low rank**, per-camera parallax/occlusion is **sparse** | Decompose each frame stack, train diffusion U-Net on $L$ (shared), per-view heads refine $S$ |
| **Long-range context eats VRAM** (tripled context) | Across time, track + cars span tiny latent subspace ⇒ **rank $\ll HW$** | Run spatio-temporal RPCA on a *(pixels × time)* matrix; store only low-rank coefficients + sparse event tensor → longer horizons per GB |
| **Dataset mis-sync noise** (two replays stitched by CV) | Mis-aligned frames appear as **column-sparse** outliers | Streaming RPCA flags columns with large $S$; drop or down-weight before training |

---

## 3  General Roles of RPCA in World Models

| Pipeline stage | RPCA action | Benefit |
|----------------|-------------|---------|
| **Frame pre-processing** | $D$ = vectorised frames; feed $L$ to model, keep $S$ as event mask | Removes flicker & jitter; isolates transients |
| **Latent-trajectory cleaning** | Stack latent states $z_t$; apply RPCA | Reduces model bias; stabilises imagination roll-outs |
| **Dataset curation** | RPCA on replay buffer; outlier episodes $\leftrightarrow$ high magnitude in $S$ | Higher sample efficiency, fewer crashes |
| **Error diagnosis** | RPCA on residuals $(o_t-\hat o_t)$ | Low-rank systematic errors vs. sparse rare failures |
| **Multi-agent** | Stack agent streams row-wise | Low-rank = shared physics; sparse = agent-specific shocks |

---

## 4  Integration Recipes

### 4.1  Pre-Processing Approach

```python
import numpy as np
from rpca import R_pca

# Example: H×W×6 frames for two cameras, T timesteps
D = frames.reshape(H*W*6, T)          # (pixels×channels) × time
L, S = R_pca(D, lmbda=1/np.sqrt(max(D.shape))).fit()

clean_frames = L.reshape(H, W, 6, T)  # input to world model
sparse_mask  = (S != 0)               # optional conditioning

Matrix layout choices
	•	• Pixels × time → long-horizon compression
	•	• Pixels × channels → cross-view coupling
```

⸻

### 4.2  Loss-Engineering Approach

$
\mathcal{L} =
\lambda_{\text{shared}} \bigl\lVert \hat L - L \bigr\rVert_2^2
;+;
\lambda_{\text{sparse}} \lVert \hat S - S \rVert_1
;+;
\mathcal{L}_{\text{diffusion / RL}}.
$

- Train backbone on $L$ (global scene).
- Lightweight heads or conditional tokens reconstruct $S$ (events).
- Anneal $\lambda_{\text{sparse}}$ after global structure is learnt.

⸻

### 4.3  Optional Nuclear-Norm Regulariser inside U-Net

Add to an intermediate feature map $F$:

$
\mathcal{L}{\text{rank}} ;=; \beta,\lVert F\rVert*,
$

nudging hidden activations toward low rank — a differentiable analogue of RPCA.

⸻



### References
	•	Candès E. J., Li X., Ma Y., Wright J. — “Robust Principal Component Analysis?” JACM 58 (3), 2011
	•	Enigma Labs blog — “Introducing Multiverse: The First AI Multiplayer World Model” (May 2025)

