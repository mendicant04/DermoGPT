# Conditional robustness of CCT at a single decoding step

This note gives the full proof of Theorem 1 in Appendix IV of the TMI revision and explicit sufficient conditions on the number of views. The [extended analysis in arXiv v1, Appendix G](https://arxiv.org/html/2601.01868v1#A7) includes the distance-only case.

## Assumptions and statement

Let $p_1,\dots,p_K$ be independent draws from $(1-\varepsilon)D_G+\varepsilon D_B$ on the probability simplex $\Delta^{V-1}$, with $0\leq\varepsilon<1/2$. Fix a target $p^\star$ and assume
$$
\mathbb{E}_{D_G}\|p-p^\star\|_2^2\leq\sigma^2,\qquad
\mathbb{E}_{D_B}\|p-p^\star\|_2^2\geq\sigma^2+\Delta^2.
$$
For some $\alpha\in(0,1)$ and $\eta\in(0,1/2)$, set $R_G=\sigma/\sqrt{\alpha}$, $R_B=\sqrt{\sigma^2+\Delta^2/2}$, and assume
$$
R_G<R_B,\qquad a=(1-\varepsilon)(1-\alpha)>1/2,\qquad
b=\|\mathbb{E}p-p^\star\|_2<\eta(R_B-R_G).
$$

Let $\bar p=K^{-1}\sum_r p_r$, $D_r=\tfrac12\|p_r-\bar p\|_2^2$, and $C_r\in[0,1]$ be the confidence margin. CCT uses $w_r\propto\exp(\lambda C_r-\beta D_r)$, with $\lambda,\beta\geq0$ and $\sum_r w_r=1$, and returns $q=\sum_r w_rp_r$.

For any $\delta\in(0,1)$ and sufficiently large $K$, with probability at least $1-\delta$ there are disjoint sets $G,B$ such that $|G|\geq\rho K$ for some $\rho>1/2$, $\|p_g-p^\star\|_2\leq R_G$ for $g\in G$, and $\|p_b-p^\star\|_2\geq R_B$ for $b\in B$. Moreover, $D_b-D_g\geq\gamma=\tfrac12(1-2\eta)(R_B^2-R_G^2)>0$ for every such pair. With $W_U=\sum_{r\notin G\cup B}w_r$,
$$
\|q-p^\star\|_2
\leq R_G+(\sqrt{2}-R_G)
\left[W_U+\frac{1-\rho}{\rho}\exp(\lambda-\beta\gamma)\right].
$$

## Proof

1. **Clean majority.** Introduce the latent mixture label of each draw. Let $G$ consist of the draws from $D_G$ with distance to $p^\star$ at most $R_G$, and let $B$ consist of the draws from $D_B$ at distance at least $R_B$. These sets are disjoint by construction. Markov's inequality gives $\Pr_{D_G}(\|p-p^\star\|_2>R_G)\leq\alpha$ for $\sigma>0$. When $\sigma=0$, a $D_G$ draw equals $p^\star$ almost surely and the same probability bound holds. The indicators for membership in $G$ are independent Bernoulli variables with mean at least $a$. For any $\rho\in(1/2,a)$, Hoeffding's inequality gives
   $$
   \Pr(|G|<\rho K)\leq\exp\!\bigl(-2K(a-\rho)^2\bigr).
   $$

2. **Barycenter proximity.** Write $\mu=\mathbb{E}p$ and $s=\eta(R_B-R_G)-b>0$. Independence and $\|p\|_2\leq1$ give $\mathbb{E}\|\bar p-\mu\|_2^2=K^{-1}(\mathbb{E}\|p\|_2^2-\|\mu\|_2^2)\leq K^{-1}$. A further application of Markov's inequality yields $\Pr(\|\bar p-\mu\|_2>s)\leq1/(Ks^2)$. Thus $\|\bar p-p^\star\|_2\leq\eta(R_B-R_G)$ holds together with the clean-majority event with probability at least $1-\delta$, for example if
   $$
   K\ \geq\ \max\!\left\{\frac{\log(2/\delta)}{2(a-\rho)^2},\,\frac{2}{\delta s^2}\right\}.
   $$

3. **Distance gap.** On this event put $d=\|\bar p-p^\star\|_2$. For $g\in G$, $\|p_g-\bar p\|_2\leq R_G+d$. For $b\in B$, $\|p_b-\bar p\|_2\geq R_B-d>0$. Squaring yields
   $$
   D_b-D_g\ \geq\ \tfrac12\big[(R_B-d)^2-(R_G+d)^2\big]
   =\tfrac12(R_B+R_G)(R_B-R_G-2d)
   \ \geq\ \gamma.
   $$

4. **Exponential weight bound.** Since $C_b-C_g\leq1$, $w_b\leq e^{\lambda-\beta\gamma}w_g$ for each $b,g$. Summing over all $b,g$ gives $W_B\leq (|B|/|G|)\,e^{\lambda-\beta\gamma}W_G$. Using $W_G\leq1$ and $|G|\geq\rho K$ gives $W_B\leq [(1-\rho)/\rho]e^{\lambda-\beta\gamma}$. The bound also holds when $B$ is empty.

5. **Aggregate error.** The simplex has diameter $\sqrt{2}$. The triangle inequality for $q-p^\star=\sum_rw_r(p_r-p^\star)$ yields
   $$
   \|q-p^\star\|_2
   \leq R_GW_G+\sqrt{2}(W_B+W_U)
   =R_G+(\sqrt{2}-R_G)(W_B+W_U).
   $$
   Substituting the bound on $W_B$ proves the claim.

The argument also covers the distance-only rule by setting $\lambda=0$. When $\beta\gamma>\lambda$, each separated contaminated view has smaller weight than every effective clean view. The result is conditional on the mixture and separation assumptions at the step under analysis. The weight on views outside the separated sets appears explicitly as $W_U$.
