# imageAlign

## 1. Introduction
The goal of image alignment is to align a template image $T$ to an 
input image $I$. This demo contains four classical image alignment 
algorithms. We implemented these algorithms in C++ language using 
[OpenCV](https://github.com/opencv/opencv/tree/3.1.0) library in 
version 3.1.0.With the aim of algorithm verification, we did not take 
the efficiency seriously, so the code can be achieve in some more 
efficient ways.

## 2. Implementation
To align a template image $T(\mathbf{x})$ to an input image 
$I(\mathbf{x})$. Our goal is to minimize:

$$
\sum_{\mathbf{x}}[I(\mathbf{W}(\mathbf{x};\mathbf{p})) - T(\mathbf{x})]^2
\tag{1}
$$

where $\mathbf{x} = (x,y)^T$ is the coordinate vector, 
$\mathbf{W}(\mathbf{x};\mathbf{p})$ is warp and 
$\mathbf{p} = (p_1, ..., p_n)^T$ is warp parameter vector. With the 
respect to $\mathbf{p}$, where the sum is performed over the pixels 
$\mathbf{x}$ in the template image $T(\mathbf{x})$.


And in this demo, we assume the warp is an affine warps:

$$
\begin{align}
    \mathbf{W}(\mathbf{x};\mathbf{p})
        & =  \begin{pmatrix}
            (1+a_{11})\cdot x &+ &a_{12}\cdot y     &+ &t_x \\
            a_{21}\cdot y     &+ &(1+a_{22})\cdot y &+ &t_y \\
            \end{pmatrix} \notag\\
        & = \begin{pmatrix}
            1+a_{11} & a_{12}   & t_x \\
            a_{21}   & 1+a_{22} & t_y \\
            \end{pmatrix}
            \begin{pmatrix}
            x \\
            y \\
            1 \\
            \end{pmatrix}
\tag{2}
\end{align}
$$

where there are 6 parameters $\mathbf{p} = (a_{11}, a_{12}, a_{21}, a_{22}, tx, ty)^T$.

- ### Forwards Additive Image Alignment

    The Lucas-Kanade Algorithm(a Guss-Newton gradient descent non-linear 
    optimization algorithm, 1981). We assumes that current estimate of 
    $\mathbf{p}$ is known. Then minimized the following expression by sloving 
    the increments $\Delta\mathbf{p}$.

    $$
    \sum_{\mathbf{x}}[I(\mathbf{W}(\mathbf{x};\mathbf{p} + \Delta\mathbf{p})) - T(\mathbf{x})]^2
    \tag{3}
    $$

    With respect to $\Delta\mathbf{p}$, and the parameters are updated:

    $$
    \mathbf{p} \leftarrow \mathbf{p} + \Delta\mathbf{p}
    \tag{4}
    $$

    These two steps are iterated until the estimates of the parameters $\mathbf{p}$ converge.
    And the way to test convergence is whether some norm of the vector $\Delta\mathbf{p}$ is 
    below a threshold $\epsilon$. i.e.$\|\Delta\mathbf{p}\| \le \epsilon$.
    
    The Lucas-Kanade Algorithm can be derived as follows. We perform a first order Tayor Expansion on Eq. $(3)$ at $\mathbf{p}$ and we get:

    $$
    \sum_{\mathbf{x}}[I(\mathbf{W}(\mathbf{x};\mathbf{p})) + \nabla I\frac{\partial \mathbf{W}}{\partial \mathbf{p}} \Delta\mathbf{p} - T(\mathbf{x})]^2 
    \tag{5}
    $$

    ***NOTE:***Where the $\nabla I = (\frac{\partial I}{\partial x},\frac{\partial I}{\partial y})$ is gradient of image $I$ evaluated at $\mathbf{W}(\mathbf{x};\mathbf{p})$,
    i.e $\nabla I = \frac{\partial I(\mathbf{W}(\mathbf{x};\mathbf{p}))}{\partial \mathbf{W}(\mathbf{x};\mathbf{p})}$.
    The $\frac{\partial \mathbf{W}}{\partial \mathbf{p}}$ is the *Jacbian* of the warp.
    For $\mathbf{W}(\mathbf{x};\mathbf{p}) = (W_x(\mathbf{x};\mathbf{p}),W_y(\mathbf{x};\mathbf{p}))^T$, then:

    $$ 
    \frac{\partial\mathbf{W}}{\partial\mathbf{p}} 
        =  \begin{pmatrix}
            \frac{\partial{W_x}}{\partial{p_1}} & \frac{\partial{W_x}}{\partial{p_2}} &... & \frac{\partial{W_x}}{\partial{p_n}}\\
            \frac{\partial{W_y}}{\partial{p_1}} & \frac{\partial{W_y}}{\partial{p_2}} &... & \frac{\partial{W_y}}{\partial{p_n}}\\
            \end{pmatrix} 
    \tag{6}
    $$

    the affine warp in Eq. $(2)$ has the *Jacobian* as follow:

    $$ 
    \frac{\partial\mathbf{W}}{\partial\mathbf{p}} =  \begin{pmatrix}
                                                        x & y & 0 & 0 & 1 & 0\\
                                                        0 & 0 & x & y & 0 & 1\\
                                                        \end{pmatrix}
    \tag{7}
    $$

    Minimizing the expression in Eq. $(5)$ is a least squares problem,
    so we get partial derivative of the Eq. $(5)$ with the respect to $\Delta\mathbf{p}$ is:

    $$
    2\sum_{\mathbf{x}}[\nabla{I}\frac{\partial\mathbf{W}}{\partial\mathbf{p}}]^T[I(\mathbf{W}(\mathbf{x};\mathbf{p})) + \nabla{I}\frac{\partial\mathbf{W}}{\partial\mathbf{p}}\Delta\mathbf{p} - T(\mathbf{x})]
    \tag{8}
    $$

    the $\nabla{I}\frac{\partial\mathbf{W}}{\partial\mathbf{p}}$ is referred as the *steepest descent images*.
    Then set this expression to zero and we get the solution for the minimum of Eq. $(5)$:

    $$
    \Delta\mathbf{p} = H^{-1}\sum_{\mathbf{x}}[T(\mathbf{x}) - \nabla{I}\frac{\partial\mathbf{W}}{\partial\mathbf{p}}]^T[I(\mathbf{W}(\mathbf{x};\mathbf{p}))]
    \tag{9}
    $$

    where $H$ is the $n \times n$ *Hessian* matrix:

    $$
    H = \sum_{\mathbf{x}}[\nabla{I}\frac{\partial\mathbf{W}}{\partial\mathbf{p}}]^T[\nabla{I}\frac{\partial\mathbf{W}}{\partial\mathbf{p}}]
    \tag{10}
    $$

    with respect to $\Delta\mathbf{p}$ and then update $\mathbf{p} \leftarrow \mathbf{p} + \Delta\mathbf{p}$.
    The corresponding to the warp is:

    $$
    \mathbf{W}(\mathbf{x};\mathbf{p}) \leftarrow \mathbf{W}(\mathbf{x};\mathbf{p} + \Delta\mathbf{p}) 
    \approx \mathbf{W}(\mathbf{x};\mathbf{p}) + \frac{\partial\mathbf{W}}{\partial\mathbf{p}}\Delta\mathbf{p}
    \tag{11}
    $$

    then we can know that the Lucas-Kanade Algorithm consists of iteratively applying Eqs. $(9)$ and $(4)$.


    >   **The Lucas-Kanade Algorithm**
    > 
    > Pre-compute 
    >> 
    (1) Compute the $\nabla I$ of image $I$

    >Iterate:
    >>
    (1) Warp $I$ with $\mathbf{W}(\mathbf{x};\mathbf{p})$ to compute $I(\mathbf{W}(\mathbf{x};\mathbf{p}))$  
    (2) Compute the error image $T(\mathbf{x}) - I(\mathbf{W}(\mathbf{x};\mathbf{p}))$  
    (3) Warp the gradient $\nabla{I}$ with $\mathbf{W}(\mathbf{x};\mathbf{p})$  
    (4) Evaluate the Jacobian $\frac{\partial\mathbf{W}}{\partial\mathbf{p}}$ at $(\mathbf{x};\mathbf{p})$  
    (5) Compute the steepest descent images $\nabla{I}\frac{\partial\mathbf{W}}{\partial\mathbf{p}}$  
    (6) Compute the Hessian matrix $H = \sum_{\mathbf{x}}[\nabla{I}\frac{\partial\mathbf{W}}{\partial\mathbf{p}}]^T[\nabla{I}\frac{\partial\mathbf{W}}{\partial\mathbf{p}}]$  
    (7) Compute $\sum_{\mathbf{x}}[\nabla{I}\frac{\partial W}{\partial p}]^T[T(\mathbf{x}) - I(\mathbf{W}(\mathbf{x};\mathbf{p}))]$  
    (8) Compute $\Delta p = H^{-1} \sum_{\mathbf{x}}[\nabla{I}\frac{\partial\mathbf{W}}{\partial\mathbf{p}}]^T[T(\mathbf{x}) - I(\mathbf{W}(\mathbf{x};\mathbf{p}))]$  
    (9) Update the parameters: $\mathbf{p} \leftarrow \mathbf p + \Delta\mathbf{p}$

    > untill $\|\Delta\mathbf{p}\| \le \epsilon$


- ### Forwards Composition Image Alignment



## Reference

- [Lucas-Kanade 20 Years On: A Unifying Framework](http://www.ncorr.com/download/publications/bakerunify.pdf)
- [Equivalence and Efficiency of Image Alignment Algorithms](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=990652)
- [Image Alignment Algorithms(Code Project)](https://www.codeproject.com/Articles/24809/Image-Alignment-Algorithms)