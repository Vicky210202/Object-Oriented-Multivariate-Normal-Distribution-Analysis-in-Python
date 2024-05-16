---
title: ' Object Oriented Multivariate Normal Distribution Analysis in Python '
tags:
  - Python
  - Multivariate analysis
  - Multivariate normal distribution
  - Correlations
  - Independence
  - Hotelling’s $T^2$
  - MANOVA
  - Wilk’s lambda
authors:
  - name: Vignesh S 
    orcid: 0000-0000-0000-0000 
    affiliation: 1     
affiliations:
 - name: Universty of Madras, India 
   index: 1
date: 16 May 2024 
bibliography: paper.bib
---


# Summary

<div style="text-align: justify;">

The multivariate normal distribution is a statistical idea that goes beyond the typical bell-shaped curve, extending to situations where we have multiple related random variables. In Python, there are some popular packages like Statsmodels, Scipy, and Pyro that deal with multivariate normal problems. However, these packages lack specific functions for tasks like finding conditional and marginal distributions, checking independence, calculating correlation coefficients, handling unequal covariance matrices in two-sample problems, conducting multivariate analysis of variance, and comparing population means using Wilk’s lambda. To address these gaps, the proposed MultiVariateNormal package aims to provide explicit functions for these tasks, making it easier for users to work with and analyze multivariate normal distributions in a more comprehensive way.

</div>

# Statement of need

<div style="text-align: justify;">

Multivariate normal distribution is the extension of univariate normal distribution $\mathcal{N}(\mu,\sigma^2)$ in which the scalar quantities $\mu$ and $\sigma^2$ is replaced with mean vector $\boldsymbol\mu$ and variance covariance matrix $\Sigma$ such that $|{\Sigma}| > 0$. Suppose that $X$ be a $(p \times 1)$ random vector, then $\mathbf{X}$ is said to follow $p$ - variate normal distribution with mean vector $\boldsymbol\mu$ and variance covariance matrix $\Sigma$. It is denoted by $\mathbf{X} \sim \mathcal{N}_p(\boldsymbol{\mu}, \Sigma)$, where $$\mathbf{X} = \begin{bmatrix} X_1 \\ X_2 \\ \vdots \\ X_p \end{bmatrix}_{p \times 1}, \quad \boldsymbol{\mu} = \begin{bmatrix} \mu_1 \\ \mu_2 \\ \vdots \\ \mu_p \end{bmatrix}_{p \times 1}\mathbf{and} \quad
\Sigma = \begin{bmatrix}
    \sigma_{1}^2 & \sigma_{12} & \ldots & \sigma_{1p} \\
    \sigma_{21} & \sigma_{2}^2 & \ldots & \sigma_{2p} \\
    \vdots & \vdots & \ddots & \vdots \\
    \sigma_{p1} & \sigma_{p2} & \ldots & \sigma_{p}^2
\end{bmatrix}_{p \times p}$$

Python by Python Software Foundation (2001) supports Object-Oriented Programming (OOP), a programming paradigm centered around the idea of objects. An algorithm for constructing a multivariate normal sample, given the sample mean vector and sample variance covariance matrix was presented by Cheng (1985). Similarly, within the package, there are three types of classes used to create instances of multivariate normal distributions along with their associated parameters $\boldsymbol\mu$ and $\Sigma$. While it's straightforward to calculate these parameters with real-time data, certain problems may provide either the variance covariance matrix or the correlation matrix alone. In such cases, creating an instance of a multivariate normal distribution with both $\boldsymbol\mu$ and $\Sigma$ becomes challenging. To address this, the package offers three types of multivariate normal objects. They are multivariate normal object with both mean vector and variance covariance matrix, multivariate normal object with only variance covariance matrix and multivariate normal object with only correlation matrix. In addition, the package includes a function that reads data explicitly, creating a multivariate normal object with both the mean vector and the variance covariance matrix. It is important to note that, with only the correlation matrix, users can compute partial and multiple correlation coefficients, with only the variance covariance matrix, users can compute all correlation coefficients and check the independence of variables and with both parameters of the distribution, users can compute marginal and conditional distributions, correlation coefficients, check independence and draw inferences on mean vectors.

</div>


# Statistics

## Marginal Distribution
<div style="text-align: justify;">

Let us consider two sets of variables $X_1, X_2, \dots, X_q$ and $X_{q + 1}, X_{q + 2}, \dots, X_p$ forming the vectors $$X^{(1)} = \begin{bmatrix}X_1 \\ X_2 \\ \vdots \\ X_q \end{bmatrix} \quad and \quad X^{(2)} = \begin{bmatrix}X_{q + 1} \\ X_{q + 2} \\ \vdots \\ X_p \end{bmatrix}$$ these variables form the random vector $$X = \begin{bmatrix}X^{(1)} \\ X^{(2)}  \end{bmatrix} = \begin{bmatrix}X_1\\ X_2\\ \vdots \\ X_{q} \\  \vdots \\ X_p \end{bmatrix}$$ 
If $\mathbf{X} \sim \mathcal{N}_p(\boldsymbol{\mu}, \Sigma)$
$$\boldsymbol{\mu} = \begin{bmatrix} \mu_1 \\ \mu_2 \\ \vdots\\ \mu_q \\ \vdots \\ \mu_p \end{bmatrix}_{p \times 1}, \quad
\Sigma = \begin{bmatrix}
    \sigma_{1}^2 & \sigma_{12} & \ldots & \sigma_{1q}  &\ldots &  \sigma_{1p} \\
    \sigma_{21} & \sigma_{2}^2 & \ldots &\sigma_{2q}  &\ldots & \sigma_{2p} \\
    \vdots & \vdots & \ddots & \vdots & &\vdots \\
    \sigma_{q1} & \sigma_{q2} & \ldots &\sigma_{q}^2  &\ldots & \sigma_{qp} \\
    \vdots & \vdots &  & \vdots& \ddots & \vdots \\
    \sigma_{p1} & \sigma_{p2} & \ldots & \sigma_{pq}  &\ldots &\sigma_{p}^2
\end{bmatrix}_{p \times p}$$
then the marginal distribution of
${X^{(1)}} \sim \mathcal{N}_q(\boldsymbol{\mu}^{(1)}, \Sigma_{11})$ and
${X^{(2)}} \sim \mathcal{N}_{p-q}(\boldsymbol{\mu}^{(2)}, \Sigma_{22})$,
where
$$\boldsymbol\mu = \begin{bmatrix} \boldsymbol\mu^{(1)} \\ \boldsymbol\mu^{(2)}\end{bmatrix} \quad and \quad \Sigma = \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22}\end{bmatrix}$$

## Conditional Distribution

The conditional distributions are of particularly simple nature because the mean vector depends only linearly on the variate held fixed and variance covariance matrix do not depend at all on the values of the fixed variates. Let $\mathbf{X} \sim \mathcal{N}_p(\boldsymbol{\mu}, \Sigma)$ and partition the random vector $\mathbf{X}$ into $q$ and $p - q$ components respectively. Then ${X^{(1)}} \sim \mathcal{N}_q(\boldsymbol{\mu}^{(1)}, \Sigma_{11})$ and
${X^{(2)}} \sim \mathcal{N}_{p-q}(\boldsymbol{\mu}^{(2)}, \Sigma_{22})$, where $$\boldsymbol\mu = \begin{bmatrix} \boldsymbol\mu^{(1)} \\ \boldsymbol\mu^{(2)}\end{bmatrix} \quad and \quad \Sigma = \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22}\end{bmatrix}$$ then the conditional distribution of $$(X^{(1)} \mid X^{(2)} = X_2)  \sim\mathcal{N}_{q}(\boldsymbol{\mu^{(1)}} +  \Sigma_{12}^{ } ~\Sigma_{22}^{-1}~(X^{(2)} - \boldsymbol{\mu}^{(2)}),\Sigma_{11.2}^{ })$$ where $$\Sigma_{11.2}^{ } = \Sigma_{11}^{ } - \Sigma_{12}^{ }~\Sigma_{22}^{-1}~\Sigma_{21}^{ }$$ Furthermore the estimate $\Sigma_{12}^{ }\Sigma_{22}^{-1}$ is known as the estimate associated with the mean vector and $\Sigma_{11.2}^{ }$ is known as the estimate of the variance covariance matrix. 

## Independence

In multivariate normal distribution, any two variables in the random vector are independent only if the covariance between them is zero. Consider a p-dimensional random vector $\mathbf{X}$, then any two variables $X_i$ and $X_j$ are independent only if $Cov(X_i, X_j) = 0$, where $i \neq j$. Instead of two univariate variables, let us consider two components of variables. These two components of variables are independent only if the covariance matrix is a zero matrix. Let $\mathbf{X} \sim \mathcal{N}_p(\boldsymbol{\mu}, \Sigma)$ and partition the random vector $\mathbf{X}$ into $q$ and $p - q$ components respectively.Then ${X^{(1)}} \sim \mathcal{N}_q(\boldsymbol{\mu}^{(1)}, \Sigma_{11})$ and ${X^{(2)}} \sim \mathcal{N}_{p-q}(\boldsymbol{\mu}^{(2)}, \Sigma_{22})$, where $$\boldsymbol\mu = \begin{bmatrix} \boldsymbol\mu^{(1)} \\ \boldsymbol\mu^{(2)}\end{bmatrix} \quad and \quad \Sigma = \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22}\end{bmatrix}$$ Here the components $X^{(1)}$ and $X^{(2)}$ are independent only if $\Sigma_{12} = O$. If we have two linear combinations of variables of random vector say $T'\mathbf{X}$ and $M'\mathbf{X}$, then the two linear combinations are independent only if
$Cov(T'\mathbf{X}, M'\mathbf{X}) = T'\Sigma M$ is zero.

## Some Important Results

1.  Let $\mathbf{X} \sim \mathcal{N}_p(\boldsymbol{\mu}, \Sigma)$ and the matrix $T$ is non-singular of order $(p \times k)$, then $T'\mathbf{X} \sim    \mathcal{N}_k(T'\boldsymbol{\mu}, T'~\Sigma~T)$

2.  Let $\mathbf{X} \sim \mathcal{N}_p(\boldsymbol{\mu}, \Sigma)$ and partition the random vector $\mathbf{X}$ into $q$ and $p - q$ components respectively. Then ${X^{(1)}} \sim \mathcal{N}_q(\boldsymbol{\mu}^{(1)}, \Sigma_{11})$ and ${X^{(2)}} \sim \mathcal{N}_{p-q}(\boldsymbol{\mu}^{(2)}, \Sigma_{22})$, where $$\boldsymbol\mu = \begin{bmatrix} \boldsymbol\mu^{(1)} \\ \boldsymbol\mu^{(2)}\end{bmatrix} \quad and \quad \Sigma = \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22}\end{bmatrix}$$ and let the matrices A and B are non singular, then $Cov(AX^{(1)}, BX^{(2)}) = A\Sigma_{12}B'$ Anderson(2003)

## Partial Correlation 

Consider a random vector $\mathbf{X}$ of dimension $(p \times 1)$, $\mathbf{X} = \begin{bmatrix} X_1, & X_2, & \ldots, & X_p \end{bmatrix}'$. The correlation between any two variables eliminating the linear effect of other variables in them is called partial correlation. The partial correlation coefficients in multivariate normal distribution are correlation coefficients in conditionals. Let $\mathbf{X} \sim \mathcal{N}_p(\boldsymbol{\mu}, \Sigma)$ and partition the random vector $\mathbf{X}$ into $2$ and $p - 2$ components respectively. Then ${X^{(1)}} \sim \mathcal{N}_2(\boldsymbol{\mu}^{(1)}, \Sigma_{11})$ and ${X^{(2)}} \sim \mathcal{N}_{p-2}(\boldsymbol{\mu}^{(2)}, \Sigma_{22})$, where $$\boldsymbol\mu = \begin{bmatrix} \boldsymbol\mu^{(1)} \\ \boldsymbol\mu^{(2)}\end{bmatrix} \quad and \quad \Sigma = \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22}\end{bmatrix}$$This partition explains that the first component contains the two variables for which we want to find the correlation keeping the remaining $p-2$
variables in the second component fixed or constant say $X_2$. Then the conditional distribution of $$(X^{(1)} \mid X^{(2)} = X_2)  \sim\mathcal{N}_{2}(\boldsymbol{\mu^{(1)}} +  \Sigma_{12}^{ } ~\Sigma_{22}^{-1}~(X^{(2)} - \boldsymbol{\mu}^{(2)}), \Sigma_{11.2}^{ })$$It is to be noted that, for finding partial correlation, it is enough to find the conditional variance covariance matrix. Moreover it is independent of the value of the variate held fixed, so $X_2$ is not mandatory to compute the partial correlation coefficient. $$\Sigma_{11.2}^{ } = \Sigma_{11}^{ } - \Sigma_{12}^{ }~\Sigma_{22}^{-1}~\Sigma_{21}^{ }$$ For this problem, the value of $$\Sigma_{11.2} = \begin{bmatrix} \sigma_{11}^o & \sigma_{12}^o \\ \sigma_{21}^o & \sigma_{22}^o  \end{bmatrix}$$ where $\sigma_{ij}^o$ is the covariance obtained by fixing the variables in $X^{(2)}$ and $i,j = 1, 2$. Then the partial coefficient is obtain from, $$\rho_{12.3 \ldots p} = \frac{\sigma_{12}^o}{\sqrt{\sigma_{11}^o~\sigma_{22}^o}}$$ There is an another way of computing the partial correlation coefficient
by using simple correlation coefficients Anderson(2003). $$\rho_{12.3 \ldots p} = \frac{\rho_{12} - \sum\limits_{i = 3}^p(\rho_{1i}\rho_{2i})}{\sqrt{(1-\sum\limits_{i=3}^p {\rho_{1i}}^2)(1-\sum\limits_{i=3}^p {\rho_{2i}}^2)}}$$

## Multiple Correlation

Multiple correlation deals with the situation in which we want the correlation between one variate and a set of variates. Let $\mathbf{X} \sim \mathcal{N}_p(\boldsymbol{\mu}, \Sigma)$ and partition the random vector $\mathbf{X}$ into $1$ and $p - 1$ components respectively.
$$\mathbf{X} = \begin{bmatrix} X_1 \\ X_2 \\ \vdots \\ X_p \end{bmatrix} = \begin{bmatrix} X^{(1)} \\ X^{(2)}\end{bmatrix},~
\boldsymbol{\mu} = \begin{bmatrix} \mu_1 \\ \mu_2 \\ \vdots \\ \mu_p \end{bmatrix} = \begin{bmatrix} \mu^{(1)} \\ \mu^{(2)} \end{bmatrix} ~\mathrm{and}~ 
\Sigma = \begin{bmatrix}
    \sigma_{1}^2 & \sigma_{12} & \ldots & \sigma_{1p} \\
    \sigma_{21} & \sigma_{2}^2 & \ldots & \sigma_{2p} \\
    \vdots & \vdots & \ddots & \vdots \\
    \sigma_{p1} & \sigma_{p2} & \ldots & \sigma_{p}^2
\end{bmatrix} = \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22}\end{bmatrix}$$
Now the multiple correlation coefficient between $X^{(1)} = X_1$ and $X^{(2)} = \begin{bmatrix} X_2, & X_3, & \ldots, & X_p \end{bmatrix}'$ defined as the maximum value of Karl Pearson correlation coefficient between $X^{(1)}$ and all possible linear combination $\alpha' X^{(2)}$, where $\alpha \in \mathbb{R}^{p-1}$. Now Consider
$$\Sigma = \begin{bmatrix}
    \sigma_{1}^2 & \sigma_{12} & \ldots & \sigma_{1p} \\
    \sigma_{21} & \sigma_{2}^2 & \ldots & \sigma_{2p} \\
    \vdots & \vdots & \ddots & \vdots \\
    \sigma_{p1} & \sigma_{p2} & \ldots & \sigma_{p}^2
\end{bmatrix} = \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22}\end{bmatrix} = \begin{bmatrix} \sigma_{11} & \sigma_{(1)} \\ \sigma_{(1)}' & \Sigma_{22}
\end{bmatrix}$$ 
then the multiple correlation coefficient is computed from $$R^2_{1.23 \ldots p} = \frac{\sigma_{(1)}^{ }~\Sigma_{22}^{-1}~\sigma_{(1)}'}{\sigma_{11}}, ~ 0 < R < 1$$ In general, let $X^{(1)} = X_i$ and $X^{(2)} = \begin{bmatrix} X_1, & X_2, & \ldots, & X_j, & \ldots, & X_p \end{bmatrix}'$ where $j \neq i$, then
$$R^2_{i.23 \ldots p} = \frac{\sigma_{(i)}^{ }~\Sigma_{22}^{-1}~\sigma_{(i)}'}{\sigma_{ii}},~ 0 < R < 1$$ 
There is an another way of computing multiple correlation coefficient by using simple correlation coefficients Anderson(2003). Suppose we have the correlation matrix for p - variate normal distribution say 
$$\rho = \begin{bmatrix}
    1 & \rho_{12} & \ldots & \rho_{1p} \\
    \rho_{21} & 1& \ldots & \rho_{2p} \\
    \vdots & \vdots & \ddots & \vdots \\
    \rho_{p1} & \rho_{p2} & \ldots & 1
\end{bmatrix}$$ 
Suppose $X^{(1)} = X_1$ and $X^{(2)} = \begin{bmatrix} X_2, & X_3, &\ldots, & X_p \end{bmatrix}'$,
then 
$$1 - R^2_{1.23 \ldots p} = \frac{\begin{vmatrix}
1 & \rho_{12} & \ldots & \rho_{1p} \\
    \rho_{21} & 1& \ldots & \rho_{2p} \\
    \vdots & \vdots & \ddots & \vdots \\
    \rho_{p1} & \rho_{p2} & \ldots & 1 \end{vmatrix}}{\begin{vmatrix}
     1& \ldots & \rho_{2p} \\
    \vdots & \ddots & \vdots \\
    \rho_{p2} & \ldots & 1
    \end{vmatrix}}$$

## One Sample Hotelling's T-Square Test 

The one sample Hotelling's $T^{ 2}$ statistic is used for testing $\mathrm{H_0} : \boldsymbol{\mu} = \boldsymbol{\mu_o}~\mathrm{Vs}~\mathrm{H_1} : \boldsymbol{\mu} \neq \boldsymbol{\mu_o}$ in $\mathcal{N}_p(\boldsymbol{\mu}, \Sigma)$, where $\Sigma$ is unknown. Let $\bar{x}$ and $S$ be the sample mean vector and variance covariance matrix computed from the given sample of size n. The one sample Hotelling's $T^{ 2}$ statistic is given by $$T^{2} =n(\bar{x} - \mu_o)'S^{-1}(\bar{x} - \mu_o)$$ At level of significance $\alpha$, under the null hypothesis $\mathrm{H_0}$ $$\frac{T^{ 2}}{n-1}\frac{n-p}{p} \sim \mathcal{F}_{p,~n~-~p}^{\alpha}$$ 

## Two Sample Hotelling's T-Square Test 

The two sample Hotelling's $T^2$ statistic is used for testing $\mathrm{H_0} : \boldsymbol{\mu_1} = \boldsymbol{\mu_2}~\mathrm{Vs}~\mathrm{H_1} : \boldsymbol{\mu_1} \neq \boldsymbol{\mu_2}$ based on $n_1$ observations from $\mathcal{N}_p(\boldsymbol{\mu_1}, \Sigma)$ and $n_2$ observations from
$\mathcal{N}_p(\boldsymbol{\mu_2}, \Sigma)$, where $\Sigma$ is the common unknown variance covariance matrix. Let $\bar{x}_1$, $S_1$ and
$\bar{x}_2$, $S_2$ be the sample mean vector and sample variance covariance matrix. Let $S$ be the pooled variance covariance matrix
which can be computed by $$S = \frac{(n_1 - 1)S_1 + (n_2 - 1)S_2}{n_1 + n_2 - 2}$$ The two sample Hotelling's $T^2$ statistic is given by $$T^2 = \frac{n_1n_2}{n_1 + n_2} (\bar{x}_1 - \bar{x}_2)'S^{-1}(\bar{x}_1 - \bar{x}_2)$$ At level of significance $\alpha$, under the null hypothesis $\mathrm{H_0}$ $$\frac{T^2}{n_1 + n_2 - 2}\frac{n_1 + n_2 - p - 1}{p} \sim \mathcal{F}_{p,~n_1~+~n_2~-~p~-~1}^{\alpha} $$ Suppose we have a situation of unequal covariance matrices, then we cannot use two sample Hotelling's $T^2$ statistic to test the equality of mean vectors Anderson(2003). This problem deals with the testing of $\mathrm{H_0} : \boldsymbol{\mu_1} = \boldsymbol{\mu_2}~\mathrm{Vs}~\mathrm{H_1} : \boldsymbol{\mu_1} \neq \boldsymbol{\mu_2}$ based on $n_1$ observations from $\mathcal{N}_p(\boldsymbol{\mu_1}, \Sigma_1)$ and $n_2$ observations from $\mathcal{N}_p(\boldsymbol{\mu_2}, \Sigma_2)$, where $\Sigma_1 \neq \Sigma_2$ but both are unknown. This problem was introduced by Fisher and later by Behren. We have two cases namely $n_1 = n_2$ and $n_1 \neq n_2$. If $n_1 = n_2$, for testing $\mathrm{H_0} : \boldsymbol{\mu_1} = \boldsymbol{\mu_2}~\mathrm{Vs}~\mathrm{H_1} : \boldsymbol{\mu_1} \neq \boldsymbol{\mu_2}$, it is equivalent to test $\mathrm{H_0} : \boldsymbol{\mu_1} - \boldsymbol{\mu_2} = 0~\mathrm{Vs}~\mathrm{H_1} : \boldsymbol{\mu_1} - \boldsymbol{\mu_2} \neq 0$. Let $\boldsymbol{\gamma} = \boldsymbol{\mu_1} - \boldsymbol{\mu_2}$. Now the null hypothesis becomes $\mathrm{H_0} : \boldsymbol{\gamma} = 0~\mathrm{Vs}~\mathrm{H_1} : \boldsymbol{\gamma} \neq 0$. Let $n_1 = n_2 = n$ and $Z_i = X_i - Y_i;~i = 1, 2, \ldots, n$, then $Z \sim \mathcal{N}_p(\boldsymbol{\gamma}, \Sigma_1 + \Sigma_2)$. Let $\bar{z}$ and $S$ be the sample mean vector and variance covariance matrix for $Z \sim \mathcal{N}_p(\boldsymbol{\gamma}, \Sigma_1 + \Sigma_2)$. The one sample Hotelling's $T^2$ for testing $\mathrm{H_0} : \boldsymbol{\gamma} = 0~\mathrm{Vs}~\mathrm{H_1} : \boldsymbol{\gamma} \neq 0$ is given by $$T^2_z =n(\bar{z})'S^{-1}(\bar{z})$$ At level of significance $\alpha$, under the null hypothesis $\mathrm{H_0}$ $$\frac{T^{ 2}_z}{n-1}\frac{n-p}{p} \sim \mathcal{F}_{p,~n~-~p}^{\alpha}$$ If $n_1 \neq n_2$, Assume $n_1 < n_2$ and $Z_i = X_i - Y_i;~i = 1,2, \ldots, n_1$, then $Z \sim \mathcal{N}_p(\boldsymbol{\gamma}, \Sigma_1 + \frac{n_1}{n_2}\Sigma_2)$. Then for testing $\mathrm{H_0} : \boldsymbol{\mu_1} = \boldsymbol{\mu_2}~\mathrm{Vs}~\mathrm{H_1} : \boldsymbol{\mu_1} \neq \boldsymbol{\mu_2}$, it is equivalent to test $\mathrm{H_0} : \boldsymbol{\mu_1} - \boldsymbol{\mu_2} = 0~\mathrm{Vs}~\mathrm{H_1} : \boldsymbol{\mu_1} - \boldsymbol{\mu_2} \neq 0$. Let $\boldsymbol{\gamma} = \boldsymbol{\mu_1} - \boldsymbol{\mu_2}$. Now the null hypothesis becomes $\mathrm{H_0} : \boldsymbol{\gamma} = 0~\mathrm{Vs}~\mathrm{H_1} : \boldsymbol{\gamma} \neq 0$.
Let $\bar{z}$ and $S$ be the sample mean vector and variance covariance matrix for $Z \sim \mathcal{N}_p(\boldsymbol{\gamma}, \Sigma_1 + \frac{n_1}{n_2}\Sigma_2)$. The one sample Hotelling's $T^2$ for testing $\mathrm{H_0} : \boldsymbol{\gamma} = 0~\mathrm{Vs}~\mathrm{H_1} : \boldsymbol{\gamma} \neq 0$
is given by $$T^2_z =n_1(\bar{z})'S^{-1}(\bar{z})$$ At level of significance $\alpha$, under the null hypothesis $\mathrm{H_0}$ $$\frac{T^{ 2}_z}{n_1-1}\frac{n_1-p}{p} \sim \mathcal{F}_{p,~n_1~-~p}^{\alpha}$$ For testing $\mathrm{H_0} : \boldsymbol{\Sigma_1} = \boldsymbol{\Sigma_2}~\mathrm{Vs}~\mathrm{H_1} : \boldsymbol{\Sigma_1} \neq \boldsymbol{\Sigma_2}$, The Box's $M$ statistic is employed which follows chi square distribution under $\mathrm{H_0}$ Johnson and Wichern (2002).

## Multivariate Analysis of Variance (MANOVA)

Often more than two populations needs to be compared. Random samples collected from each of $g$ populations are arranged as $$\begin{array}{ccccc} Population_1 :  &  X_{11} & X_{12} & \ldots & X_{1n_1}  \\ Population_2 :  &  X_{21} & X_{22} & \ldots & X_{2n_2}  \\ \vdots&  \vdots & \vdots & \vdots & \vdots \\ Population_g :  &  X_{g1} & X_{g2} & \ldots & X_{gn_g}  \\ \end{array}$$ Has some assumptions that,

1.  $X_{l1},X_{l2}, \ldots, X_{ln_l}$ is a random sample of size $n_l$
    from a population with mean
    $\mu_l,~\mathrm{for}~l = 1, 2, \ldots, g$.

2.  The random samples from different populations are independent.

3.  All population have a common covariance matrix $\Sigma$.

4.  Each population is a multivariate normal.

The MANOVA model for comparing mean vectors is $$X_{lj} = \mu + {\tau}_l + e_{lj}\quad j = 1,~2,~\ldots~n_l~\mathrm{and}~l = 1,~2,~\ldots~g.$$ The sum of squares and cross product matrix and its associated degrees of freedom is summarized in the below MANOVA table.

<centre>

  | Source of variation | Sum of squares and cross product | Degrees of freedom |
|---------------------|----------------------------------|--------------------|
| Treatment           | $B = \sum\limits_{l = 1}^gn_l(\bar{x}_l - \bar{x})(\bar{x}_l - \bar{x})'$ | $g - 1$ |
| Residual            | $W = \sum\limits_{l = 1}^g\sum\limits_{j = 1}^{n_l}(x_{lj} - \bar{x}_l)(x_{lj} - \bar{x}_l)'$ | $\sum\limits_{l = 1}^gn_l - g$ |
| Total               | $B + W = \sum\limits_{l = 1}^g\sum\limits_{j = 1}^{n_l}(x_{lj} - \bar{x})(x_{lj} - \bar{x})'$ | $\sum\limits_{l = 1}^gn_l - 1$ |

</centre>

<centre> One-way MANOVA Table </centre>

For testing
$\mathrm{H_0}:{\tau}_1 = {\tau}_2 = \cdots = {\tau}_g$
Vs
$\mathrm{H_1}:{\tau}_i \neq {\tau}_j\mathrm{~(atleast~one)~for~}i \neq j$.
The Wilk's lambda test statistic is employed and it is given by
$$\lambda^* = \frac{|W|}{|B + W|} = \frac{\left|\sum\limits_{l = 1}^g\sum\limits_{j = 1}^{n_l}(x_{lj} - \bar{x}_l)(x_{lj} - \bar{x}_l)'\right|}{\left|\sum\limits_{l = 1}^g\sum\limits_{j = 1}^{n_l}(x_{lj} - \bar{x})(x_{lj} - \bar{x})' \right|}$$
At level of significance $\alpha$, under the null hypothesis
$\mathrm{H_0}$
$$F = \frac{1 - \sqrt[b]{\lambda^{*}}}{\sqrt[b]{\lambda^{*}}}\frac{ab - c}{p(g - 1)} \sim \mathcal{F}_{p(g~-~1),~ab~-~c}^{\alpha}$$
where $$a = \sum\limits_{l = 1}^gn_l - g - \frac{p - g + 2}{2}$$ $$b = \left\{ \begin{array}{cr} \sqrt{\frac{p^2(g - 1)^2 - 4}{p^2 + (g - 1)^2 + 5}} &  ~\mathrm{if}~p^2 + (g - 1)^2 + 5 > 0  \\ 1  &  ;~\mathrm{if}~p^2 + (g - 1)^2 + 5 \leq 0 \end{array}\right.$$ $$c = \frac{p(g - 1) - 2}{2}$$

The stabilized probability plot, with approximately equal variances of plotted points prompted the definition of a new goodness-of-fit statistic DSP. This statistic aids in constructing acceptance regions for QQ, pp, and the new plots, reducing subjectivity in interpretation were discussed by Michael (1983) and this inspired the creation of a specialized plot for visualizing hypothesis test results. The Critical and Acceptance Region plot is used for this purpose, providing a visual interpretation of the test outcomes. The concepts of hypothesis testing were referred from Rohatgi et al. (2002) and Lehmann (1986).

</div>

# Citations
- Anderson, T. W. (2003). *An Introduction to Multivariate Statistical Analysis* (3rd ed.). John Wiley & Sons.
- Johnson, R. A., & Wichern, D. W. (2002). *Applied Multivariate Statistical Analysis* (5th ed.). Prentice Hall.
- Cheng, R. C. H. (1985). "Generation of multivariate normal samples with given sample mean and covariance matrix." *Journal of Statistical Computation and Simulation*, 21(1), 39–49. [DOI: 10.1080/00949658508810795](https://doi.org/10.1080/00949658508810795)
- Michael, J. R. (1983). "The stabilized probability plot." *Biometrika*, 70(1), 11–17. [DOI: 10.1093/biomet/70.1.11](https://doi.org/10.1093/biomet/70.1.11)
- Rohatgi, V. K., & others. (2002). *An Introduction to Probability and Statistics*. John Wiley.
- Lehmann, E. L. (1986). *Testing of Statistical Hypotheses*. John Wiley.

# References
- [Python Software Foundation. (2001). *Python Software, Version 3.11.7*.](https://www.python.org)
