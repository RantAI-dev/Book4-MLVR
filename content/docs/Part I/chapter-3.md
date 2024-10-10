---
weight: 1000
title: "Chapter 3"
description: "Mathematics for Machine Learning"
icon: "article"
date: "2024-10-10T22:52:03.193779+07:00"
lastmod: "2024-10-10T22:52:03.193779+07:00"
katex: true
draft: false
toc: true
---
{{% alert icon="💡" context="info" %}}
<strong>"<em>Pure mathematics is, in its way, the poetry of logical ideas.</em>" — Albert Einstein</strong>
{{% /alert %}}

{{% alert icon="📘" context="success" %}}
<p style="text-align: justify;"><em>Chapter 3 of MLVR provides a robust foundation in the essential mathematical concepts needed to understand and implement machine learning algorithms. The chapter covers a range of topics, starting with linear algebra, which forms the backbone of many machine learning techniques, and extending to calculus and optimization, which are critical for training models. It also delves into probability and statistics, providing the tools to model uncertainty and make inferences from data. Additionally, the chapter explores linear models and least squares, numerical methods for solving complex problems, and discrete mathematics and graph theory, which are crucial for structuring and analyzing data in machine learning. Each section combines theoretical explanations with practical Rust implementations, ensuring that readers not only grasp the mathematical concepts but also learn how to apply them effectively in their machine learning projects.</em></p>
{{% /alert %}}

# 3.1. Linear Algebra in Machine Learning
<p style="text-align: justify;">
Linear algebra is the cornerstone of machine learning. It provides the mathematical framework for representing and manipulating data, as well as for developing algorithms that learn from data. Understanding vectors, matrices, tensors, and their operations is essential for anyone looking to delve deep into machine learning, especially when implementing algorithms in a systems programming language like Rust.
</p>

<p style="text-align: justify;">
A vector in $\mathbb{R}^n$ is an ordered collection of $n$ real numbers. It can be represented as a column vector:
</p>

<p style="text-align: justify;">
$$ \mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} $$
</p>
<p style="text-align: justify;">
Vectors are used to represent data points, features, and parameters in machine learning models. They support operations such as addition, scalar multiplication, and dot product.
</p>

<p style="text-align: justify;">
A matrix is a two-dimensional array of numbers with mmm rows and nnn columns, denoted as $\mathbf{A} \in \mathbb{R}^{m \times n}$:
</p>

<p style="text-align: justify;">
$$ \mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \dots & a_{1n} \\ a_{21} & a_{22} & \dots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \dots & a_{mn} \end{bmatrix} $$
</p>
<p style="text-align: justify;">
Matrices represent datasets, linear transformations, and coefficients in systems of linear equations. Key matrix operations are:
</p>

- <p style="text-align: justify;">Addition:</p>
- <p style="text-align: justify;">Matrices of the same dimension can be added element-wise: $(\mathbf{A} + \mathbf{B})_{ij} = a_{ij} + b_{ij}$.</p>
- <p style="text-align: justify;">Multiplication:</p>
- <p style="text-align: justify;">Scalar Multiplication: Multiplying a matrix by a scalar $\alpha$: $(\alpha \mathbf{A})_{ij} = \alpha a_{ij}$ .</p>
- <p style="text-align: justify;">Matrix Multiplication: The product of an $m \times n$ matrix $\mathbf{A}$ and an $n \times p$ matrix $\mathbf{B}$is an $m \times p$ matrix $\mathbf{C}$: $c_{ik} = \sum_{j=1}^{n} a_{ij} b_{jk}$ .</p>
- <p style="text-align: justify;">Transposition:</p>
- <p style="text-align: justify;">The transpose of a matrix $\mathbf{A}$, denoted $\mathbf{A}^\top$, is obtained by swapping its rows and columns: $(\mathbf{A}^\top)_{ij} = a_{ji}$ .</p>
<p style="text-align: justify;">
A vector space $V$ over a field $F$ is a set equipped with two operations: vector addition and scalar multiplication, satisfying certain axioms (associativity, distributivity, etc.). Examples include $\mathbb{R}^n$ and the space of continuous functions.
</p>

<p style="text-align: justify;">
A basis of a vector space $V$ is a set of linearly independent vectors $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$ that span $V$. Any vector $\mathbf{v} \in V$ can be uniquely expressed as a linear combination of the basis vectors:
</p>

<p style="text-align: justify;">
$$ \mathbf{v} = \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \dots + \alpha_k \mathbf{v}_k $$
</p>
<p style="text-align: justify;">
The rank of a matrix $\mathbf{A}$ is the maximum number of linearly independent column vectors (column rank) or row vectors (row rank). It represents the dimension of the column space or row space of $\mathbf{A}$.
</p>

<p style="text-align: justify;">
Vectors $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$ are linearly independent if the only solution to:
</p>

<p style="text-align: justify;">
$$ \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \dots + \alpha_k \mathbf{v}_k = \mathbf{0} $$
</p>
<p style="text-align: justify;">
is $\alpha_1 = \alpha_2 = \dots = \alpha_k = 0$. Linear independence is crucial in determining the basis and rank of a vector space.
</p>

<p style="text-align: justify;">
A tensor is a multidimensional array that generalizes scalars (0th-order tensors), vectors (1st-order tensors), and matrices (2nd-order tensors). An $n$-th order tensor can be represented as $T \in \mathbb{R}^{I_1 \times I_2 \times \dots \times I_n}$, where $I_k$ is the dimension along the $k$-th axis.
</p>

<p style="text-align: justify;">
Tensor operations include:
</p>

- <p style="text-align: justify;">Tensor Addition: Element-wise addition of tensors with the same dimensions.</p>
- <p style="text-align: justify;">Tensor Multiplication: Generalizations of matrix multiplication, such as tensor dot products and outer products.</p>
- <p style="text-align: justify;">Tensor Contraction: Summing over specific indices, reducing the tensor's order.</p>
<p style="text-align: justify;">
Matrix decompositions are techniques to factorize matrices into products of matrices with specific properties, facilitating easier computations and insights into the matrix's structure.
</p>

<p style="text-align: justify;">
LU Decomposition factors a square matrix $\mathbf{A}$ into the product of a lower triangular matrix $\mathbf{L}$ and an upper triangular matrix $\mathbf{U}$:
</p>

<p style="text-align: justify;">
$$ \mathbf{A} = \mathbf{L}\mathbf{U} $$
</p>
<p style="text-align: justify;">
This decomposition is useful for solving linear systems $\mathbf{A}\mathbf{x} = \mathbf{b}$, computing determinants, and inverting matrices.
</p>

<p style="text-align: justify;">
Singular Value Decomposition (SVD) is a fundamental matrix factorization technique that decomposes any $m \times n$ matrix $\mathbf{A}$ into three specific matrices: $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$. In this decomposition:
</p>

- <p style="text-align: justify;">$\mathbf{U} \in \mathbb{R}^{m \times m}$ is an orthogonal matrix whose columns are the left singular vectors of $\mathbf{A}$.</p>
- <p style="text-align: justify;">$\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$ is a diagonal matrix containing the singular values of $\mathbf{A}$, which are non-negative real numbers arranged in descending order along the diagonal.</p>
- <p style="text-align: justify;">$\mathbf{V}^\top \in \mathbb{R}^{n \times n}$ is the transpose of an orthogonal matrix $\mathbf{V}$, whose columns are the right singular vectors of $\mathbf{A}$.</p>
<p style="text-align: justify;">
SVD is instrumental in various applications such as data compression, noise reduction, and dimensionality reduction techniques like Principal Component Analysis (PCA). By decomposing a matrix into its singular values and vectors, SVD allows us to approximate the original matrix with a lower-rank matrix, capturing the most significant features of the data while discarding less important information.
</p>

<p style="text-align: justify;">
Given a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, the concepts of eigenvalues and eigenvectors become particularly relevant. A scalar $\lambda$ is called an eigenvalue of $\mathbf{A}$, and a non-zero vector $\mathbf{v}$ is its corresponding eigenvector if they satisfy the equation $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$. Eigenvalues and eigenvectors reveal intrinsic properties of the linear transformation represented by $\mathbf{A}$, such as scaling factors and invariant directions. These properties are essential for understanding the behavior of systems and transformations in various fields, including machine learning.
</p>

<p style="text-align: justify;">
In machine learning, techniques like PCA and Linear Discriminant Analysis (LDA) heavily rely on the concepts of SVD, eigenvalues, and eigenvectors for dimensionality reduction and feature extraction.
</p>

- <p style="text-align: justify;">Principal Component Analysis (PCA) reduces data dimensionality by projecting the data onto principal components—the directions of maximum variance. This involves computing the eigenvectors and eigenvalues of the data's covariance matrix. The eigenvectors (principal components) corresponding to the largest eigenvalues capture the most significant variance in the data. By projecting the data onto these principal components, PCA simplifies the dataset while preserving its essential structure and patterns.</p>
- <p style="text-align: justify;">Linear Discriminant Analysis (LDA) seeks linear combinations of features that best separate different classes within the data. It does this by maximizing the ratio of between-class variance to within-class variance, leading to an eigenvalue problem involving scatter matrices. Solving this problem yields discriminant vectors that, when used to project the data, enhance class separability. LDA is particularly useful in supervised classification tasks where the goal is to improve the model's ability to distinguish between classes.</p>
- <p style="text-align: justify;">Dimensionality Reduction techniques like PCA and LDA are crucial for managing high-dimensional datasets common in machine learning. By reducing the number of features while retaining significant structural information, these methods improve computational efficiency and help mitigate the curse of dimensionality—a phenomenon where the feature space becomes so vast that the available data becomes sparse, making it difficult for algorithms to learn effectively. Dimensionality reduction enhances model performance by focusing on the most informative aspects of the data, reducing noise, and preventing overfitting.</p>
<p style="text-align: justify;">
Implementing mathematical concepts like linear algebra in Rust necessitates efficient and reliable handling of complex numerical computations. Rust, known for its performance and safety, offers libraries such as <code>nalgebra</code> and <code>ndarray</code> that provide robust data structures and functions for linear algebra operations. These libraries enable developers to implement algorithms like matrix decompositions, Principal Component Analysis (PCA), and Linear Discriminant Analysis (LDA) effectively.
</p>

<p style="text-align: justify;">
<code>nalgebra</code> is a comprehensive linear algebra library for Rust that supports vectors, matrices, and a wide range of operations on them. It is designed to be generic and efficient, making it suitable for high-performance applications in machine learning and scientific computing.
</p>

<p style="text-align: justify;">
In this example, we'll demonstrate how to perform basic matrix and vector operations using <code>nalgebra</code>, such as defining matrices and vectors and performing matrix-vector multiplication.
</p>

{{< prism lang="rust" line-numbers="true">}}
use nalgebra::{Matrix3, Vector3};

fn main() {
    // Define a 3x3 matrix
    let a = Matrix3::new(
        1.0, 2.0, 3.0,  // First row
        4.0, 5.0, 6.0,  // Second row
        7.0, 8.0, 9.0,  // Third row
    );

    // Define a 3-dimensional vector
    let v = Vector3::new(1.0, 0.0, -1.0);

    // Perform matrix-vector multiplication
    let result = a * v;

    // Display the result
    println!("Result of A * v:\n{}", result);
}
{{< /prism >}}
<p style="text-align: justify;">
In the code above, we first import the <code>Matrix3</code> and <code>Vector3</code> types from the <code>nalgebra</code> crate. We then define a $3 \times 3$ matrix <code>a</code> using the <code>Matrix3::new</code> method, providing nine elements that correspond to the entries of the matrix in row-major order. Each row of the matrix is represented by three consecutive elements in the <code>new</code> method. Next, we define a 3-dimensional vector <code>v</code> using <code>Vector3::new</code>, specifying its components along the x, y, and z axes.
</p>

<p style="text-align: justify;">
We perform matrix-vector multiplication using the <code><em></code> operator, which is overloaded in <code>nalgebra</code> to handle this operation. The expression <code>a </em> v</code> computes the product of the matrix <code>a</code> and the vector <code>v</code>, resulting in a new vector <code>result</code> that represents the transformed coordinates. Finally, we print the result to the console using <code>println!</code>, allowing us to verify the correctness of the operation.
</p>

<p style="text-align: justify;">
This basic operation is fundamental in machine learning algorithms where data points (vectors) are often transformed or projected using matrices. It demonstrates how Rust, with the help of <code>nalgebra</code>, can efficiently handle linear algebra computations that are essential for implementing machine learning models.
</p>

<p style="text-align: justify;">
Matrix decompositions are essential in solving linear systems, optimizing computations, and understanding the properties of matrices in machine learning algorithms. <code>nalgebra</code> provides functionalities to perform various matrix decompositions, including LU decomposition and Singular Value Decomposition (SVD).
</p>

<p style="text-align: justify;">
LU decomposition factors a square matrix into the product of a lower triangular matrix and an upper triangular matrix. It is particularly useful for solving systems of linear equations and inverting matrices.
</p>

{{< prism lang="rust" line-numbers="true">}}
use nalgebra::{Matrix3, Vector3, LU};

fn main() {
    // Define a 3x3 matrix
    let a = Matrix3::new(
         2.0,  1.0, 1.0,
         4.0, -6.0, 0.0,
        -2.0,  7.0, 2.0,
    );

    // Perform LU decomposition
    let lu = LU::new(a);

    // Check if the matrix is invertible (non-singular)
    if lu.is_invertible() {
        // Define the right-hand side vector
        let b = Vector3::new(5.0, -2.0, 9.0);

        // Solve the linear system Ax = b
        let x = lu.solve(&b).expect("Solution not found");

        // Display the solution vector x
        println!("Solution x:\n{}", x);
    } else {
        println!("Matrix is singular and cannot be decomposed.");
    }
}
{{< /prism >}}
<p style="text-align: justify;">
In this example, we begin by defining a $3 \times 3$ matrix <code>a</code> with specific values that ensure the matrix is non-singular (invertible). The matrix is created using <code>Matrix3::new</code>, and the elements are provided in row-major order. We then perform LU decomposition on this matrix by calling <code>LU::new(a)</code>, which computes the decomposition and returns an <code>LU</code> object.
</p>

<p style="text-align: justify;">
We check whether the matrix is invertible using <code>lu.is_invertible()</code>. If the matrix is invertible, we proceed to solve the system of linear equations <code>Ax = b</code>. We define the vector <code>b</code> representing the right-hand side of the equation using <code>Vector3::new</code>. The <code>lu.solve(&b)</code> method efficiently computes the solution vector <code>x</code> by utilizing the previously computed LU decomposition. We handle any potential errors using <code>expect</code>, which will panic with the message "Solution not found" if the system cannot be solved.
</p>

<p style="text-align: justify;">
Finally, we print the solution vector <code>x</code> to the console. If the matrix is singular (non-invertible), we output an appropriate message indicating that the matrix cannot be decomposed. This example demonstrates how LU decomposition simplifies solving linear systems, which is a common requirement in various machine learning algorithms where parameters are estimated by solving equations.
</p>

<p style="text-align: justify;">
SVD decomposes a matrix into its singular values and singular vectors, which are instrumental in data analysis techniques like PCA.
</p>

{{< prism lang="rust" line-numbers="true">}}
use nalgebra::{DMatrix, SVD};

fn main() {
    // Define a 4x3 data matrix
    let data = DMatrix::from_row_slice(4, 3, &[
        1.0, 0.0, 0.0,  // First row
        0.0, 2.0, 0.0,  // Second row
        0.0, 0.0, 3.0,  // Third row
        1.0, 2.0, 3.0,  // Fourth row
    ]);

    // Perform Singular Value Decomposition
    let svd = SVD::new(data.clone(), true, true);

    // Extract the singular values and singular vectors
    let singular_values = svd.singular_values;
    let u = svd.u.unwrap();       // Left singular vectors
    let v_t = svd.v_t.unwrap();   // Right singular vectors (transposed)

    // Display the results
    println!("Singular values:\n{}", singular_values);
    println!("Left singular vectors U:\n{}", u);
    println!("Right singular vectors V^T:\n{}", v_t);
}
{{< /prism >}}
<p style="text-align: justify;">
We start by defining a data matrix <code>data</code> with dimensions 4x3 using <code>DMatrix::from_row_slice</code>. This function allows us to create a dynamic matrix (whose size is determined at runtime) by providing the number of rows, number of columns, and a slice of elements in row-major order. The matrix represents a dataset with 4 samples and 3 features.
</p>

<p style="text-align: justify;">
Next, we perform Singular Value Decomposition on the data matrix by calling <code>SVD::new(data.clone(), true, true)</code>. The <code>clone</code> method is used to pass a copy of the data matrix since <code>SVD::new</code> may consume the input. The boolean arguments indicate whether to compute the full <code>u</code> and <code>v_t</code> matrices (the left and right singular vectors).
</p>

<p style="text-align: justify;">
We extract the singular values from <code>svd.singular_values</code>, which is a vector containing the singular values in descending order. We also retrieve the left singular vectors <code>u</code> and the right singular vectors transposed <code>v_t</code> by unwrapping the optional values <code>svd.u</code> and <code>svd.v_t</code>. These matrices are essential components in the SVD.
</p>

<p style="text-align: justify;">
Finally, we print the singular values and the singular vectors to the console. This SVD implementation is crucial in reducing dimensionality, compressing data, and identifying underlying patterns, which are essential steps in algorithms like PCA. By decomposing the data matrix into its singular values and vectors, we can understand the structure of the data and perform operations like noise reduction and feature extraction.
</p>

<p style="text-align: justify;">
Principal Component Analysis (PCA) is a statistical procedure that transforms data into a new coordinate system, reducing the dimensionality while retaining the most variance.
</p>

{{< prism lang="rust" line-numbers="true">}}
use nalgebra::{DMatrix, DVector, SymmetricEigen};

fn pca(x: &DMatrix<f64>, n_components: usize) -> (DMatrix<f64>, DVector<f64>, DMatrix<f64>) {
    // Step 1: Standardize the data (mean = 0)
    let mean = x.column_mean();
    let repeated_mean = DMatrix::from_fn(x.nrows(), x.ncols(), |i, _| mean[i]);
    let centered_data = x - repeated_mean;

    // Step 2: Compute the covariance matrix
    let covariance_matrix = (centered_data.transpose() * &centered_data) / (x.nrows() as f64 - 1.0);

    // Step 3: Perform eigenvalue decomposition
    let eigen = SymmetricEigen::new(covariance_matrix);

    // Step 4: Sort eigenvalues and eigenvectors in descending order
    let mut eigen_pairs: Vec<(f64, DVector<f64>)> = eigen
        .eigenvalues
        .iter()
        .zip(eigen.eigenvectors.column_iter())
        .map(|(&val, vec)| (val, vec.clone_owned()))
        .collect();

    eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Step 5: Select the top n_components eigenvectors (principal components)
    let principal_components = DMatrix::from_columns(&eigen_pairs.iter()
        .take(n_components)
        .map(|(_, vec)| vec.clone())
        .collect::<Vec<_>>());

    // Step 6: Transform the data onto the new subspace
    let transformed_data = centered_data * &principal_components;

    (transformed_data, DVector::from_vec(eigen_pairs.iter().map(|(val, _)| *val).collect()), principal_components)
}

fn main() {
    // Sample data: 5 samples, 3 features
    let data = DMatrix::from_row_slice(5, 3, &[
        2.5, 2.4, 0.5,
        0.5, 0.7, -0.1,
        2.2, 2.9, 0.7,
        1.9, 2.2, 0.3,
        3.1, 3.0, 1.1,
    ]);

    // Apply PCA
    let n_components = 2;
    let (transformed_data, eigenvalues, components) = pca(&data, n_components);

    // Display the results
    println!("Transformed data (PCA):\n{}", transformed_data);
    println!("Eigenvalues:\n{}", eigenvalues);
    println!("Principal Components:\n{}", components);
}
{{< /prism >}}
<p style="text-align: justify;">
We begin by defining a data matrix <code>data</code> with 5 samples and 3 features using <code>DMatrix::from_row_slice</code>. Each row in the matrix represents a sample, and each column represents a feature. The dataset might represent measurements or observations in a machine learning context.
</p>

<p style="text-align: justify;">
In the first step, we center the data by computing the mean of each feature using <code>data.column_mean()</code> and subtracting it from the corresponding feature values. This operation ensures that the dataset has a mean of zero along each feature, which is a prerequisite for PCA.
</p>

<p style="text-align: justify;">
Next, we compute the covariance matrix of the centered data using the formula $\text{Cov} = \frac{1}{n - 1} X^\top X$ where $X$ is the centered data matrix and nnn is the number of samples. The covariance matrix captures the variance and covariance between different features, providing insights into how they vary together.
</p>

<p style="text-align: justify;">
We then perform eigenvalue decomposition on the covariance matrix using <code>SymmetricEigen::new(cov_matrix)</code>. This function computes the eigenvalues and eigenvectors of the symmetric covariance matrix. The eigenvalues represent the amount of variance captured by each principal component, and the eigenvectors indicate the directions of these components in the feature space.
</p>

<p style="text-align: justify;">
After obtaining the eigenvalues and eigenvectors, we pair them together and sort them in descending order based on the eigenvalues using the <code>sort_by</code> method. This step ensures that the principal components are ordered according to the amount of variance they explain.
</p>

<p style="text-align: justify;">
We select the top <code>k</code> eigenvectors to form the principal components, effectively reducing the dimensionality of the data from the original number of features to <code>k</code>. In this example, we select the top two principal components.
</p>

<p style="text-align: justify;">
Finally, we transform the centered data onto the new subspace defined by the principal components using matrix multiplication. The result is the transformed data with reduced dimensions, retaining the most significant variance from the original dataset. We print the transformed data to observe the outcome of the PCA.
</p>

<p style="text-align: justify;">
This implementation of PCA in Rust demonstrates how to reduce dimensionality, which is beneficial for visualization, noise reduction, and improving the efficiency of machine learning algorithms. By projecting the data onto the principal components, we simplify the dataset while preserving its essential structure and patterns.
</p>

<p style="text-align: justify;">
Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction technique that maximizes class separability.
</p>

{{< prism lang="rust" line-numbers="true">}}
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use std::collections::HashMap;

fn main() {
    // Sample data matrix: 8 samples, 2 features
    let data = DMatrix::from_row_slice(8, 2, &[
        4.0, 2.0,  // Sample 1 - Class 0
        2.0, 4.0,  // Sample 2 - Class 0
        2.0, 3.0,  // Sample 3 - Class 0
        3.0, 6.0,  // Sample 4 - Class 0
        6.0, 2.0,  // Sample 5 - Class 1
        7.0, 3.0,  // Sample 6 - Class 1
        8.0, 5.0,  // Sample 7 - Class 1
        7.0, 4.0,  // Sample 8 - Class 1
    ]);

    let labels = vec![0, 0, 0, 0, 1, 1, 1, 1];

    // Compute the mean vectors for each class
    let mut class_means = HashMap::new();
    let mut class_counts = HashMap::new();

    for (i, &label) in labels.iter().enumerate() {
        let sample = data.row(i).transpose();
        class_means
            .entry(label)
            .and_modify(|mean: &mut DVector<f64>| *mean += &sample)
            .or_insert(sample.clone());
        class_counts.entry(label).and_modify(|c| *c += 1).or_insert(1);
    }

    for (label, mean) in class_means.iter_mut() {
        *mean /= *class_counts.get(label).unwrap() as f64;
        println!("Mean vector for class {}: {}", label, mean);
    }

    // Compute the within-class scatter matrix Sw
    let mut sw = DMatrix::zeros(2, 2);
    for (i, &label) in labels.iter().enumerate() {
        let sample = data.row(i).transpose();
        let mean = class_means.get(&label).unwrap();
        let diff = &sample - mean;
        sw += &diff * diff.transpose();
        println!("Diff for sample {} (class {}): {}", i, label, diff);
    }

    println!("Within-class scatter matrix Sw:\n{}", sw);

    // Compute the between-class scatter matrix Sb
    let overall_mean = data.row_mean();
    println!("Overall mean vector: {}", overall_mean);

    let mut sb = DMatrix::zeros(2, 2);
    for (&label, mean) in &class_means {
        let n = *class_counts.get(&label).unwrap() as f64;
        let mean_diff = mean - &overall_mean.transpose();

        println!("Mean diff:\n{}\nMean diff T:\n{}\nSB:\n{}", mean_diff, mean_diff.transpose(), sb);
        sb += n * &mean_diff * mean_diff.transpose(); 
        println!("Mean diff for class {}: {}", label, mean_diff);
    }

    println!("Between-class scatter matrix Sb:\n{}", sb);

    // Solve the generalized eigenvalue problem for Sb and Sw
    let sw_inv = sw.try_inverse().expect("Within-class scatter matrix is singular");
    let mat = sw_inv * sb;
    let eigen = SymmetricEigen::new(mat);

    println!("Eigenvalues: {}", eigen.eigenvalues);
    println!("Eigenvectors:\n{}", eigen.eigenvectors);

    // Select the eigenvector with the largest eigenvalue
    let max_eigenvalue_index = eigen.eigenvalues.imax();
    let w = eigen.eigenvectors.column(max_eigenvalue_index).clone_owned();

    println!("Selected eigenvector (lda direction): {}", w);

    // Project the data onto the new LDA component
    let projected_data = data * w;

    // Display the projected data
    println!("Projected data:\n{}", projected_data);
}
{{< /prism >}}
<p style="text-align: justify;">
In this example, we implement LDA to reduce the dimensionality of a dataset while maximizing class separability. We start by defining a data matrix <code>data</code> with 8 samples and 2 features using <code>DMatrix::from_row_slice</code>. Each sample belongs to one of two classes, indicated by the <code>labels</code> vector, where class 0 and class 1 represent different categories.
</p>

<p style="text-align: justify;">
We compute the mean vector for each class by iterating over the samples and accumulating the sum of the feature vectors for each class. We use a <code>HashMap</code> to associate each class label with its mean vector and keep track of the number of samples in each class. After summing the samples, we divide by the class counts to obtain the mean vectors.
</p>

<p style="text-align: justify;">
Next, we compute the within-class scatter matrix <code>Sw</code>, which measures the scatter of samples around their respective class means. We calculate <code>Sw</code> by summing the outer products of the differences between each sample and its class mean.
</p>

<p style="text-align: justify;">
We then compute the between-class scatter matrix <code>Sb</code>, which measures the scatter of the class means around the overall mean of the dataset. The overall mean is calculated using <code>data.transpose().row_mean()</code>. We compute <code>Sb</code> by summing the outer products of the differences between each class mean and the overall mean, weighted by the number of samples in each class.
</p>

<p style="text-align: justify;">
To find the optimal projection vector that maximizes class separability, we solve the generalized eigenvalue problem $Sw^{-1} Sb w = \lambda w$. We compute the inverse of <code>Sw</code> using <code>try_inverse()</code>, which may fail if <code>Sw</code> is singular. Multiplying <code>Sw^{-1}</code> and <code>Sb</code> gives us the matrix <code>mat</code> on which we perform eigenvalue decomposition using <code>SymmetricEigen::new(mat)</code>.
</p>

<p style="text-align: justify;">
We select the eigenvector corresponding to the largest eigenvalue, which represents the direction that maximizes the separation between the classes. This eigenvector <code>w</code> is the optimal linear discriminant.
</p>

<p style="text-align: justify;">
Finally, we project the original data onto <code>w</code> using matrix multiplication. The resulting <code>projected_data</code> is a one-dimensional representation of the data that enhances the distinction between the two classes. We print the projected data to observe the effectiveness of the dimensionality reduction and class separation.
</p>

<p style="text-align: justify;">
This LDA implementation demonstrates how to reduce dimensionality while preserving class discriminatory information, which is valuable in supervised learning tasks such as classification. By projecting the data onto the linear discriminant, we simplify the dataset and potentially improve the performance of classification algorithms.
</p>

<p style="text-align: justify;">
<code>ndarray</code> is a Rust library that provides N-dimensional array objects similar to NumPy's arrays in Python. It supports various operations on tensors, making it suitable for deep learning and advanced numerical computations.
</p>

#### **Example:** Creating and Manipulating Tensors
{{< prism lang="rust" line-numbers="true">}}
use ndarray::{Array3, Axis};

fn main() {
    // Create a 3x3x3 tensor (3-dimensional array) filled with zeros
    let mut tensor = Array3::<f64>::zeros((3, 3, 3));

    // Set values in the tensor based on indices
    for ((i, j, k), elem) in tensor.indexed_iter_mut() {
        *elem = (i + j + k) as f64;
    }

    // Access a slice of the tensor along a specific axis
    let slice = tensor.index_axis(Axis(2), 1);

    // Display the slice
    println!("Tensor slice at index 1 along axis 2:\n{:?}", slice);

    // Perform element-wise operations (e.g., squaring each element)
    let tensor_squared = &tensor * &tensor;

    // Display the squared tensor
    println!("Tensor squared:\n{:?}", tensor_squared);
}
{{< /prism >}}
<p style="text-align: justify;">
In this example, we demonstrate how to create and manipulate tensors using the <code>ndarray</code> crate. We begin by creating a three-dimensional array (tensor) <code>tensor</code> of size $3 \times 3 \times 3$ filled with zeros using <code>Array3::<f64>::zeros((3, 3, 3))</code>. The <code>Array3</code> type represents a three-dimensional array with elements of type <code>f64</code>.
</p>

<p style="text-align: justify;">
We populate the tensor with values based on their indices by iterating over the array using <code>indexed_iter_mut()</code>. This method provides both the indices <code>(i, j, k)</code> and mutable references to the elements <code>elem</code>. We assign to each element the sum of its indices converted to a floating-point number. This operation fills the tensor with values that reflect their position within the array.
</p>

<p style="text-align: justify;">
Next, we access a two-dimensional slice of the tensor along a specific axis using <code>index_axis(Axis(2), 1)</code>. This method extracts all elements where the third index is <code>1</code>, effectively selecting the middle "slice" of the tensor along the third dimension. We store this slice in the variable <code>slice</code> and print it to the console to visualize a cross-section of the tensor.
</p>

<p style="text-align: justify;">
We perform an element-wise operation by squaring each element of the tensor using the expression <code>&tensor * &tensor</code>. This operation leverages the <code>ndarray</code> library's ability to perform arithmetic operations on arrays in an element-wise fashion. We store the result in <code>tensor_squared</code> and print it to observe the outcome.
</p>

<p style="text-align: justify;">
This example illustrates basic tensor manipulations, which are fundamental in implementing deep learning models where multi-dimensional data and operations are prevalent. Tensors are used to represent data such as images, sequences, and feature maps in neural networks. The <code>ndarray</code> library provides the tools necessary to handle these complex data structures efficiently in Rust.
</p>

<p style="text-align: justify;">
In summary, a deep understanding of linear algebra is essential for machine learning. Vectors, matrices, and tensors are fundamental structures for representing data and computations. Matrix decompositions like LU and SVD enable efficient solutions to linear systems and reveal important properties of transformations. Implementing those mathematical concepts in Rust leverages the language's performance and safety advantages. Libraries like <code>nalgebra</code> and <code>ndarray</code> provide powerful tools for handling complex numerical computations efficiently.
</p>

- <p style="text-align: justify;">By utilizing <code>nalgebra</code>, developers can perform linear algebra operations, decompositions, and transformations essential in machine learning algorithms. The code examples demonstrate how to define matrices and vectors, perform matrix-vector multiplication, and implement advanced techniques like LU decomposition and SVD.</p>
- <p style="text-align: justify;">With <code>ndarray</code>, developers can manipulate multi-dimensional arrays (tensors), supporting operations required in deep learning and high-dimensional data analysis. The example showcases how to create tensors, manipulate their elements, and perform element-wise operations.</p>
<p style="text-align: justify;">
The detailed explanations provided bridge the gap between theoretical understanding and practical application in machine learning via Rust. By understanding and utilizing these libraries, ML developers can implement advanced machine learning techniques in Rust, contributing to high-performance applications that benefit from Rust's concurrency and memory safety features.
</p>

# 3.2. Calculus and Optimization
<p style="text-align: justify;">
Calculus and optimization are foundational pillars in machine learning, providing the mathematical tools necessary for modeling, analyzing, and solving complex problems. Calculus enables us to understand how functions change, while optimization techniques allow us to find the best parameters for our models by minimizing or maximizing certain functions, often referred to as loss or cost functions.
</p>

<p style="text-align: justify;">
Differentiation and integration are the two main operations in calculus. Differentiation concerns the rate at which a function changes, providing us with the derivative, which represents the slope or gradient of the function at any given point. For a real-valued function $f(x)$, the derivative $f'(x)$ is defined as:
</p>

<p style="text-align: justify;">
$$ f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x} $$
</p>
<p style="text-align: justify;">
This limit, if it exists, gives the instantaneous rate of change of the function with respect to $x$. Differentiation is crucial in machine learning for finding the minima or maxima of loss functions by analyzing how small changes in parameters affect the output.
</p>

<p style="text-align: justify;">
Integration, on the other hand, is the mathematical operation that aggregates or accumulates quantities over an interval. It can be considered the inverse operation of differentiation. The definite integral of a function $f(x)$ over an interval $[a, b]$ is given by:
</p>

<p style="text-align: justify;">
$$\int_a^b f(x) \, dx$$
</p>
<p style="text-align: justify;">
In machine learning, integration is less commonly used than differentiation but plays a role in probabilistic models and expectations.
</p>

<p style="text-align: justify;">
When dealing with functions of multiple variables, partial derivatives measure how the function changes with respect to one variable while keeping the others constant. For a function $f(x, y)$, the partial derivatives with respect to $x$ and $y$ are:
</p>

<p style="text-align: justify;">
$$ \frac{\partial f}{\partial x} = \lim_{\Delta x \to 0} \frac{f(x + \Delta x, y) - f(x, y)}{\Delta x} $$
</p>
<p style="text-align: justify;">
$$ \frac{\partial f}{\partial y} = \lim_{\Delta y \to 0} \frac{f(x, y + \Delta y) - f(x, y)}{\Delta y} $$
</p>
<p style="text-align: justify;">
Partial derivatives are essential in optimizing functions of several variables, which is the typical case in machine learning models with multiple parameters.
</p>

<p style="text-align: justify;">
The chain rule is a fundamental theorem for computing the derivative of composite functions. If a function $z$ depends on $y$, which in turn depends on xxx (i.e., $z = f(y)$ and $y = g(x)$, then the derivative of $z$ with respect to $x$ is:
</p>

<p style="text-align: justify;">
$$ \frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx} $$
</p>
<p style="text-align: justify;">
In the context of functions with multiple variables, the multivariable chain rule extends to partial derivatives. The chain rule is crucial in backpropagation algorithms in neural networks, where we compute gradients through layers of composed functions.
</p>

<p style="text-align: justify;">
The gradient of a scalar-valued function $f(\mathbf{x})$ with respect to a vector $\mathbf{x} = [x_1, x_2, \dots, x_n]^T$ is a vector of partial derivatives:
</p>

<p style="text-align: justify;">
$$ \nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix} $$
</p>
<p style="text-align: justify;">
The gradient points in the direction of the steepest increase of the function. In optimization problems, particularly in machine learning, we often seek to minimize a loss function, so we move in the opposite direction of the gradient (the direction of steepest descent). Gradient computation is fundamental in training machine learning models, as it tells us how to adjust parameters to reduce the loss.
</p>

<p style="text-align: justify;">
Gradient descent is an iterative optimization algorithm used to find the minimum of a function. Starting from an initial point, we take steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point. The update rule for the parameter vector $\boldsymbol{\theta}$ is:
</p>

<p style="text-align: justify;">
$$ \boldsymbol{\theta}^{(k+1)} = \boldsymbol{\theta}^{(k)} - \alpha \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}^{(k)}) $$
</p>
<p style="text-align: justify;">
Here, $\alpha$ is the learning rate, a hyperparameter that determines the size of the steps taken in the parameter space. $J(\boldsymbol{\theta})$ is the loss function, and $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}^{(k)})$ is the gradient of the loss with respect to $\boldsymbol{\theta}$ at iteration $k$.
</p>

<p style="text-align: justify;">
Gradient descent is widely used in machine learning to minimize loss functions by iteratively updating the model parameters in the direction that most rapidly decreases the loss.
</p>

<p style="text-align: justify;">
A loss function quantifies the discrepancy between the predicted outputs of a model and the actual target values. It serves as a measure of how well the model is performing. Common loss functions include:
</p>

- <p style="text-align: justify;">Mean Squared Error (MSE) for regression: $J(\boldsymbol{\theta}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$</p>
- <p style="text-align: justify;">Cross-Entropy Loss for classification: $J(\boldsymbol{\theta}) = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$</p>
<p style="text-align: justify;">
Where $y_i$ are the true labels, $\hat{y}_i$ are the predicted outputs, and $n$ is the number of samples.
</p>

<p style="text-align: justify;">
The choice of loss function depends on the problem at hand and influences the optimization process.
</p>

<p style="text-align: justify;">
Gradients indicate how to adjust the parameters to decrease the loss most effectively. By computing the gradient of the loss function with respect to the parameters, we obtain the direction in which the loss increases most rapidly. Moving in the opposite direction (negative gradient) reduces the loss. This is the core idea behind gradient descent and other optimization algorithms used in training machine learning models.
</p>

<p style="text-align: justify;">
In neural networks, backpropagation is used to compute gradients efficiently by applying the chain rule backward through the network layers.
</p>

<p style="text-align: justify;">
Convexity is a property of functions where any line segment between two points on the function lies above or on the graph. A function $f$ is convex if:
</p>

<p style="text-align: justify;">
$$ f(\lambda x + (1 - \lambda) y) \leq \lambda f(x) + (1 - \lambda) f(y) $$
</p>
<p style="text-align: justify;">
for all $x, y$ in the domain and $\lambda \in [0, 1]$.
</p>

<p style="text-align: justify;">
In convex optimization problems, any local minimum is also a global minimum. This property simplifies the optimization process because gradient descent is guaranteed to converge to the global minimum given appropriate conditions. In non-convex problems, like training deep neural networks, multiple local minima and saddle points exist, making the optimization landscape more complex and the convergence to a global minimum not guaranteed.
</p>

<p style="text-align: justify;">
Understanding convexity helps in selecting appropriate optimization algorithms and in analyzing the convergence properties of these algorithms.
</p>

<p style="text-align: justify;">
To concretize these concepts, we'll implement gradient descent to find the minimum of a simple quadratic function $f(x) = x^2$. The gradient of this function is $f'(x) = 2x$.
</p>

{{< prism lang="rust" line-numbers="true">}}
fn main() {
    let mut x = 10.0; // Initial guess
    let learning_rate = 0.1;
    let max_iterations = 1000;
    let tolerance = 1e-6;

    for iteration in 0..max_iterations {
        let gradient = 2.0 * x; // Derivative of f(x) = x^2
        let new_x = x - learning_rate * gradient;

        if f64::abs(new_x - x) < tolerance {
            println!("Converged after {} iterations.", iteration);
            break;
        }

        x = new_x;
    }

    println!("The minimum value of f(x) occurs at x = {:.6}", x);
}
{{< /prism >}}
<p style="text-align: justify;">
In this Rust program, we initialize the variable <code>x</code> with a starting value of <code>10.0</code>. We set a learning rate of <code>0.1</code>, which controls the step size in each iteration. The <code>max_iterations</code> and <code>tolerance</code> variables are used to control the stopping criteria.
</p>

<p style="text-align: justify;">
Inside the loop, we calculate the gradient of the function at the current <code>x</code> by computing <code>2.0 * x</code>. We update <code>x</code> by moving in the opposite direction of the gradient, scaled by the learning rate. The loop continues until the absolute change in <code>x</code> is less than the specified tolerance, indicating convergence.
</p>

<p style="text-align: justify;">
When the algorithm converges, it prints the number of iterations taken and the approximate value of <code>x</code> at which the minimum occurs. This example demonstrates the fundamental mechanism of gradient descent in finding the minimum of a differentiable function.
</p>

<p style="text-align: justify;">
Backpropagation is an algorithm used to compute the gradients of the loss function with respect to the weights of a neural network. It applies the chain rule to propagate the error backward through the network layers.
</p>

<p style="text-align: justify;">
Let's implement a simple single-layer neural network with one input, one output, and a sigmoid activation function.
</p>

{{< prism lang="rust" line-numbers="true">}}
use std::f64::consts::E;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

fn main() {
    // Input data and target output
    let x = 0.5;
    let target = 1.0;

    // Initial weight
    let mut weight = 0.5;
    let learning_rate = 0.1;

    // Training loop
    for _ in 0..1000 {
        // Forward pass
        let net_input = weight * x;
        let output = sigmoid(net_input);

        // Compute loss (mean squared error)
        let loss = 0.5 * (output - target).powi(2);

        // Backward pass (compute gradients)
        let error = output - target;
        let grad_output = error * sigmoid_derivative(net_input);
        let grad_weight = grad_output * x;

        // Update weight
        weight -= learning_rate * grad_weight;

        // Optional: Check for convergence (omitted for brevity)
    }

    println!("Trained weight: {:.6}", weight);
}
{{< /prism >}}
<p style="text-align: justify;">
In this code, we define the <code>sigmoid</code> function and its derivative, which are commonly used in neural networks. We initialize the input <code>x</code> and the target output <code>target</code>. The weight <code>weight</code> is initialized to <code>0.5</code>, and we set a learning rate of <code>0.1</code>.
</p>

<p style="text-align: justify;">
Within the training loop, we perform the forward pass by computing the net input to the neuron (<code>net_input</code>) and then applying the sigmoid activation function to obtain the output. The loss is computed using the mean squared error between the network output and the target.
</p>

<p style="text-align: justify;">
In the backward pass, we compute the error (<code>output - target</code>) and then calculate the gradient of the loss with respect to the net input (<code>grad_output</code>). We apply the chain rule, multiplying the error by the derivative of the sigmoid function. The gradient of the weight (<code>grad_weight</code>) is then obtained by multiplying <code>grad_output</code> by the input <code>x</code>.
</p>

<p style="text-align: justify;">
We update the weight by subtracting the product of the learning rate and the gradient from the current weight. After training, we print the trained weight, which should have adjusted to minimize the loss function.
</p>

<p style="text-align: justify;">
This example illustrates the core concepts of backpropagation and how gradients are computed and used to update the weights in a neural network.
</p>

<p style="text-align: justify;">
We will now implement linear regression using gradient descent to minimize the mean squared error loss function. The goal is to find the optimal weight and bias that best fit the data.
</p>

{{< prism lang="rust" line-numbers="true">}}
use ndarray::{Array1, Zip};

fn main() {
    // Sample data (inputs and targets)
    let x_data = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
    let y_data = Array1::from(vec![2.0, 4.0, 6.0, 8.0]);

    // Initialize weight and bias
    let mut weight = 0.0;
    let mut bias = 0.0;
    let learning_rate = 0.01;
    let epochs = 1000;

    for _ in 0..epochs {
        // Forward pass: compute predictions
        let predictions = x_data.mapv(|x| x * weight + bias);

        // Compute the errors
        let errors = &predictions - &y_data;

        // Compute the loss (mean squared error)
        let loss = errors.mapv(|e| f64::powi(e, 2)).mean().unwrap();

        // Compute gradients
        let grad_weight = 2.0 * Zip::from(&errors)
            .and(&x_data)
            .map_collect(|&e, &x| e * x)
            .mean()
            .unwrap();
        let grad_bias = 2.0 * errors.mean().unwrap();

        // Update parameters
        weight -= learning_rate * grad_weight;
        bias -= learning_rate * grad_bias;
    }

    println!("Trained weight: {:.6}", weight);
    println!("Trained bias: {:.6}", bias);
}
{{< /prism >}}
<p style="text-align: justify;">
We use the <code>ndarray</code> crate to handle arrays efficiently. The input data <code>x_data</code> and target data <code>y_data</code> are defined as <code>Array1</code> objects. We initialize the <code>weight</code> and <code>bias</code> to zero and set a learning rate and the number of epochs.
</p>

<p style="text-align: justify;">
In each epoch, we perform the following steps:
</p>

- <p style="text-align: justify;">Forward Pass: Compute the predictions by applying the current <code>weight</code> and <code>bias</code> to the input data.</p>
- <p style="text-align: justify;">Compute Errors: Calculate the difference between the predictions and the actual target values.</p>
- <p style="text-align: justify;">Compute Loss: Calculate the mean squared error loss by squaring the errors and computing their mean.</p>
- <p style="text-align: justify;">Compute Gradients:</p>
- <p style="text-align: justify;">The gradient with respect to the weight (<code>grad_weight</code>) is computed by multiplying the errors with the input data, taking the mean, and scaling by 2.0.</p>
- <p style="text-align: justify;">The gradient with respect to the bias (<code>grad_bias</code>) is computed by taking the mean of the errors and scaling by 2.0.</p>
- <p style="text-align: justify;">Update Parameters: Adjust the <code>weight</code> and <code>bias</code> by subtracting the product of the learning rate and their respective gradients.</p>
<p style="text-align: justify;">
After training, we print the trained <code>weight</code> and <code>bias</code>. Since the data follows the relationship $y = 2x$, we expect the trained weight to be close to <code>2.0</code> and the bias to be close to <code>0.0</code>.
</p>

<p style="text-align: justify;">
This example demonstrates how gradient descent can be used to optimize loss functions in machine learning models. It shows the practical implementation of linear regression in Rust, reinforcing the theoretical concepts discussed earlier.
</p>

<p style="text-align: justify;">
Calculus and optimization are essential tools in machine learning, providing the mathematical foundation for understanding and improving models. Differentiation and partial derivatives allow us to compute gradients, which indicate how to adjust parameters to minimize loss functions. The chain rule is fundamental in backpropagation algorithms, enabling efficient computation of gradients in complex models like neural networks.
</p>

<p style="text-align: justify;">
Gradient descent and its variants are the backbone of optimization algorithms in machine learning. They rely on gradients to iteratively update model parameters, moving towards a minimum of the loss function. Understanding the role of convexity helps in analyzing the convergence properties of these optimization algorithms.
</p>

<p style="text-align: justify;">
Implementing these concepts in Rust leverages the language's performance and safety features, making it suitable for high-performance computing tasks in machine learning. The provided code examples illustrate how to implement gradient descent, backpropagation, and loss function optimization from scratch using Rust and its numerical computing libraries like <code>ndarray</code>.
</p>

<p style="text-align: justify;">
By combining rigorous mathematical theory with practical implementation, we gain a deeper understanding of how machine learning models learn from data and how we can optimize them effectively. This knowledge is crucial for developing efficient and robust machine learning applications using Rust.
</p>

# 3.3. Probability and Statistics
<p style="text-align: justify;">
Probability and statistics are fundamental components of machine learning, providing the mathematical framework for modeling uncertainty, making inferences from data, and predicting future outcomes. They enable us to handle real-world data, which is often noisy and uncertain, by quantifying variability and uncertainty in a principled way. Understanding these concepts is essential for developing robust machine learning models that can generalize well to unseen data.
</p>

<p style="text-align: justify;">
Probability theory is the mathematical study of randomness and uncertainty. It quantifies the likelihood of events occurring, using a scale from 0 (impossible event) to 1 (certain event). The foundational rules of probability allow us to compute the probabilities of complex events based on simpler ones.
</p>

<p style="text-align: justify;">
For mutually exclusive events $A$ and $B$ (events that cannot occur simultaneously), the addition rule states that the probability of either event occurring is the sum of their individual probabilities:
</p>

<p style="text-align: justify;">
$$ P(A \text{ or } B) = P(A) + P(B) $$
</p>
<p style="text-align: justify;">
For independent events $A$ and $B$ (the occurrence of one does not affect the other), the multiplication rule indicates that the probability of both events occurring together is the product of their probabilities:
</p>

<p style="text-align: justify;">
$$ P(A \text{ and } B) = P(A) \times P(B) $$
</p>
<p style="text-align: justify;">
The complement rule relates the probability of an event occurring to the probability of it not occurring:
</p>

<p style="text-align: justify;">
$$ P(\text{not } A) = 1 - P(A) $$
</p>
<p style="text-align: justify;">
These fundamental principles enable the calculation of probabilities for more complex scenarios by breaking them down into simpler, constituent events.
</p>

<p style="text-align: justify;">
A random variable is a function that assigns numerical values to the outcomes of a random process or experiment. It serves as a bridge between abstract probability concepts and real-world quantities. Random variables can be classified into two main types:
</p>

- <p style="text-align: justify;">Discrete Random Variables: These take on a countable set of distinct values. Examples include the number of heads in ten coin tosses or the number of defective items in a batch.</p>
- <p style="text-align: justify;">Continuous Random Variables: These take on an uncountable set of values within an interval. Examples include the exact height of individuals in a population or the time it takes for a computer to process a task.</p>
<p style="text-align: justify;">
The behavior of a random variable is described by its probability distribution, which specifies the likelihood of each possible value or range of values it can assume.
</p>

<p style="text-align: justify;">
Probability distributions characterize how probabilities are distributed over the values of a random variable. They are essential for modeling and analyzing random phenomena in machine learning.
</p>

<p style="text-align: justify;">
Normal Distribution (Gaussian Distribution): This continuous distribution is characterized by its mean $\mu$ and variance $\sigma^2$. Its probability density function (PDF) is given by:
</p>

<p style="text-align: justify;">
$$ f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right) $$
</p>
<p style="text-align: justify;">
The normal distribution is fundamental due to the Central Limit Theorem, which states that the sum of a large number of independent, identically distributed random variables tends toward a normal distribution, regardless of their original distributions. This property makes it widely applicable in statistics and machine learning.
</p>

<p style="text-align: justify;">
Binomial Distribution: This discrete distribution models the number of successes in $n$ independent Bernoulli trials (experiments with two possible outcomes, like success/failure) with probability $p$ of success. Its probability mass function (PMF) is:
</p>

<p style="text-align: justify;">
$$ P(k) = \binom{n}{k} p^k (1 - p)^{n - k} $$
</p>
<p style="text-align: justify;">
where $\binom{n}{k}$ is the binomial coefficient representing the number of ways to choose $k$ successes out of nnn trials.
</p>

<p style="text-align: justify;">
Other important distributions include the Uniform, Poisson, and Exponential distributions, each modeling different types of random phenomena encountered in machine learning and statistical analyses.
</p>

<p style="text-align: justify;">
These are fundamental statistical measures that describe the properties of random variables:
</p>

- <p style="text-align: justify;">Expectation (Mean): The expected value $E[X]$ of a random variable $X$ is the long-run average value it assumes over numerous trials. For discrete random variables:</p>
<p style="text-align: justify;">
$$ E[X] = \sum_{i} x_i P(X = x_i) $$
</p>
- <p style="text-align: justify;">For continuous random variables:</p>
<p style="text-align: justify;">
$$E[X] = \int_{-\infty}^{\infty} x f(x) \, dx $$
</p>
- <p style="text-align: justify;">where $f(x)$ is the PDF of $X$.</p>
- <p style="text-align: justify;">Variance: The variance $\text{Var}(X)$ measures the spread or dispersion of a random variable around its mean:</p>
<p style="text-align: justify;">
$$ \text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2 $$
</p>
- <p style="text-align: justify;">A higher variance indicates that the data points are more spread out from the mean.</p>
- <p style="text-align: justify;">Covariance: The covariance $\text{Cov}(X, Y)$ measures how two random variables $X$ and $Y$ change together:</p>
<p style="text-align: justify;">
$$ \text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y] $$
</p>
- <p style="text-align: justify;">A positive covariance indicates that $X$ and $Y$ tend to increase together, while a negative covariance suggests that when one increases, the other tends to decrease.</p>
<p style="text-align: justify;">
Bayes' theorem provides a way to update our beliefs about the probability of an event $A$ based on new evidence $B$. It links the prior probability $P(A)$ with the likelihood $P(B | A)$ and the marginal probability $P(B)$ to compute the posterior probability $P(A | B)$:
</p>

<p style="text-align: justify;">
$$ P(A | B) = \frac{P(B | A) P(A)}{P(B)} $$
</p>
<p style="text-align: justify;">
This theorem is foundational in Bayesian inference, allowing us to revise our predictions or hypotheses in light of new data. In machine learning, it underpins algorithms that incorporate prior knowledge and continuously update predictions as more information becomes available.
</p>

<p style="text-align: justify;">
For continuous random variables, the probability density function $f(x)$ describes the relative likelihood of the variable taking on a specific value. The PDF must satisfy the following properties:
</p>

- <p style="text-align: justify;">$f(x) \geq 0$ for all $x$.</p>
- <p style="text-align: justify;">The total area under the PDF curve is 1: $\int_{-\infty}^{\infty} f(x) \, dx = 1$</p>
- <p style="text-align: justify;">The probability that $X$ falls within an interval $[a, b]$ is given by:</p>
<p style="text-align: justify;">
$$P(a \leq X \leq b) = \int_{a}^{b} f(x) \, dx$$
</p>
<p style="text-align: justify;">
The PDF provides a complete description of the distribution of a continuous random variable, enabling the calculation of probabilities and expectations.
</p>

<p style="text-align: justify;">
Statistical methods are integral to various machine learning tasks:
</p>

- <p style="text-align: justify;">Classification: Algorithms like the Naive Bayes classifier use probabilistic models to predict the class labels of input data based on feature probabilities. These models assume certain statistical properties about the data, such as feature independence.</p>
- <p style="text-align: justify;">Hypothesis Testing: Statistical tests are used to determine whether there is enough evidence to reject a null hypothesis. In machine learning, this might involve testing whether a new model performs significantly better than a baseline model.</p>
- <p style="text-align: justify;">Inference: Estimating the parameters of probability distributions that best explain the observed data is a common task. Methods like Maximum Likelihood Estimation (MLE) and Bayesian inference are used to fit models to data.</p>
<p style="text-align: justify;">
Statistics provides the tools to make informed decisions under uncertainty, quantify the confidence in predictions, and validate models through rigorous testing.
</p>

<p style="text-align: justify;">
Rust's emphasis on performance and safety makes it well-suited for implementing probabilistic models. Below is an example of implementing a simple Naive Bayes classifier for binary classification.
</p>

<p style="text-align: justify;">
The Naive Bayes classifier applies Bayes' theorem with the assumption of independence between features. It calculates the posterior probability of each class given the input features and predicts the class with the highest probability.
</p>

{{< prism lang="rust" line-numbers="true">}}
use std::collections::HashMap;

#[derive(Debug)]
struct NaiveBayesClassifier {
    class_priors: HashMap<String, f64>,
    feature_likelihoods: HashMap<String, HashMap<String, f64>>,
}

impl NaiveBayesClassifier {
    // Training function to compute priors and likelihoods
    fn train(data: Vec<(Vec<String>, String)>) -> Self {
        let mut class_counts = HashMap::new();
        let mut feature_counts = HashMap::new();
        let mut total_samples = 0.0;

        for (features, class_label) in &data {
            *class_counts.entry(class_label.clone()).or_insert(0.0) += 1.0;
            total_samples += 1.0;

            for feature in features {
                let feature_map = feature_counts
                    .entry(class_label.clone())
                    .or_insert_with(HashMap::new);
                *feature_map.entry(feature.clone()).or_insert(0.0) += 1.0;
            }
        }

        // Compute class priors
        let mut class_priors = HashMap::new();
        for (class_label, count) in &class_counts {
            class_priors.insert(class_label.clone(), count / total_samples);
        }

        // Compute feature likelihoods with Laplace smoothing
        let mut feature_likelihoods = HashMap::new();
        for (class_label, features_map) in &feature_counts {
            let total_features = features_map.values().sum::<f64>() + features_map.len() as f64;
            let mut likelihoods = HashMap::new();
            for (feature, count) in features_map {
                likelihoods.insert(feature.clone(), (count + 1.0) / total_features);
            }
            feature_likelihoods.insert(class_label.clone(), likelihoods);
        }

        NaiveBayesClassifier {
            class_priors,
            feature_likelihoods,
        }
    }

    // Prediction function to classify new instances
    fn predict(&self, features: Vec<String>) -> String {
        let mut max_log_prob = f64::MIN;
        let mut best_class = String::new();

        for (class_label, &prior) in &self.class_priors {
            let mut log_prob = prior.ln();

            if let Some(feature_probs) = self.feature_likelihoods.get(class_label) {
                for feature in &features {
                    if let Some(&likelihood) = feature_probs.get(feature) {
                        log_prob += likelihood.ln();
                    } else {
                        // Assign a small probability to unseen features
                        log_prob += (1e-6_f64).ln();
                    }
                }
            }

            if log_prob > max_log_prob {
                max_log_prob = log_prob;
                best_class = class_label.clone();
            }
        }

        best_class
    }
}

fn main() {
    // Training data: (features, class_label)
    let training_data = vec![
        (vec!["sunny".into(), "hot".into(), "high".into(), "weak".into()], "no".into()),
        (vec!["sunny".into(), "hot".into(), "high".into(), "strong".into()], "no".into()),
        (vec!["overcast".into(), "hot".into(), "high".into(), "weak".into()], "yes".into()),
        (vec!["rain".into(), "mild".into(), "high".into(), "weak".into()], "yes".into()),
        // Additional samples can be added here
    ];

    let classifier = NaiveBayesClassifier::train(training_data);

    // Test data
    let test_features = vec!["sunny".into(), "cool".into(), "high".into(), "strong".into()];
    let prediction = classifier.predict(test_features);

    println!("Predicted class: {}", prediction);
}
{{< /prism >}}
<p style="text-align: justify;">
In this implementation, we define a <code>NaiveBayesClassifier</code> struct that stores the prior probabilities of each class and the likelihoods of each feature given the class. The <code>train</code> method processes the training data to compute these probabilities. It counts the occurrences of each class and each feature within each class, then calculates the class priors and feature likelihoods, applying Laplace smoothing to handle unseen features.
</p>

<p style="text-align: justify;">
The <code>predict</code> method takes a new set of features and calculates the log-probabilities for each class to prevent underflow from multiplying many small probabilities. It sums the log of the class prior and the logs of the feature likelihoods. The class with the highest log-probability is selected as the prediction.
</p>

<p style="text-align: justify;">
In the <code>main</code> function, we create some training data and use the classifier to predict the class of a test instance. This example demonstrates how probabilistic models, specifically the Naive Bayes classifier, can be implemented in Rust for classification tasks.
</p>

<p style="text-align: justify;">
Bayesian inference allows us to update our beliefs about a parameter or hypothesis in light of new evidence. Let's consider estimating the probability $\theta$ that a biased coin lands heads. We use a Beta distribution as the prior and update it with observed data from coin tosses.
</p>

{{< prism lang="rust" line-numbers="true">}}
use rand_distr::{Beta, Distribution};
use rand::thread_rng;

fn main() {
    // Prior parameters (Beta distribution)
    let mut alpha = 1.0; // Prior count of heads (successes)
    let mut beta_param = 1.0; // Prior count of tails (failures)

    // Observed data: sequence of coin tosses
    let observations = vec!['H', 'T', 'H', 'H', 'T', 'H', 'T', 'H', 'H', 'T'];

    // Update parameters based on observations
    for outcome in observations {
        match outcome {
            'H' => alpha += 1.0,
            'T' => beta_param += 1.0,
            _ => (),
        }
    }

    // Posterior distribution is Beta(alpha, beta_param)
    println!("Posterior parameters: alpha = {:.1}, beta = {:.1}", alpha, beta_param);

    // Generate a probability estimate from the posterior
    let beta_dist = Beta::new(alpha, beta_param).unwrap();
    let mut rng = thread_rng();
    let sample_theta = beta_dist.sample(&mut rng);

    println!("Estimated probability of heads: {:.4}", sample_theta);
}
{{< /prism >}}
<p style="text-align: justify;">
In this example, we start with a prior belief about the probability θ\\thetaθ of the coin landing heads, represented by a Beta(1, 1) distribution (uniform prior). We then update this prior using observed data from coin tosses, incrementing the <code>alpha</code> parameter for each head and the <code>beta_param</code> parameter for each tail. The resulting posterior distribution reflects our updated belief about $\theta$ after considering the data.
</p>

<p style="text-align: justify;">
We use the <code>rand</code> crate to work with the Beta distribution and sample an estimated value of $\theta$ from the posterior. This value represents a plausible estimate of the true probability of the coin landing heads, given our prior and the observed data.
</p>

<p style="text-align: justify;">
This code demonstrates how Bayesian inference is applied to update beliefs and estimate parameters based on evidence, a fundamental concept in probabilistic modeling and machine learning.
</p>

<p style="text-align: justify;">
Rust can also be utilized for statistical analyses such as hypothesis testing and confidence interval calculation. Below is an implementation of a two-sample t-test to compare the means of two independent samples.
</p>

{{< prism lang="rust" line-numbers="true">}}
use statrs::statistics::Statistics;
use statrs::distribution::{ContinuousCDF, StudentsT};

fn main() {
    // Sample data from two groups
    let sample1 = vec![2.3, 1.9, 2.5, 2.7, 2.0];
    let sample2 = vec![1.8, 1.6, 2.1, 1.5, 1.7];

    // Compute sample means and variances
    let mean1 = sample1.clone().mean();
    let mean2 = sample2.clone().mean();
    let var1 = sample1.clone().variance();
    let var2 = sample2.clone().variance();
    let n1 = sample1.clone().len() as f64;
    let n2 = sample2.clone().len() as f64;

    // Compute pooled standard deviation
    let pooled_var = (((n1 - 1.0) * var1) + ((n2 - 1.0) * var2)) / (n1 + n2 - 2.0);
    let pooled_std = pooled_var.sqrt();

    // Compute t-statistic
    let t_stat = (mean1 - mean2) / (pooled_std * ((1.0 / n1 + 1.0 / n2).sqrt()));

    // Degrees of freedom
    let df = n1 + n2 - 2.0;

    // Compute p-value
    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));

    // Output the results
    println!("Sample 1 Mean: {:.4}", mean1);
    println!("Sample 2 Mean: {:.4}", mean2);
    println!("t-statistic: {:.4}", t_stat);
    println!("Degrees of freedom: {:.2}", df);
    println!("p-value: {:.4}", p_value);

    // Decision based on p-value
    if p_value < 0.05 {
        println!("Reject the null hypothesis: Means are significantly different.");
    } else {
        println!("Fail to reject the null hypothesis: No significant difference between means.");
    }
}
{{< /prism >}}
<p style="text-align: justify;">
In this implementation, we use the <code>statrs</code> crate for statistical computations. We have two independent samples, <code>sample1</code> and <code>sample2</code>, representing measurements from two groups. We calculate the means and variances of both samples.
</p>

<p style="text-align: justify;">
The pooled standard deviation combines the variances of the two samples, accounting for their sizes. Using this pooled standard deviation, we compute the t-statistic, which measures the difference between the sample means relative to the variability in the data.
</p>

<p style="text-align: justify;">
The degrees of freedom are calculated based on the sample sizes. We then compute the p-value using the cumulative distribution function (CDF) of the Student's t-distribution. The p-value indicates the probability of observing a t-statistic as extreme as the one calculated, assuming the null hypothesis (that the means are equal) is true.
</p>

<p style="text-align: justify;">
Based on the p-value and a significance level of 0.05, we decide whether to reject the null hypothesis. This statistical test helps determine if there is a significant difference between the two groups, which is essential in experimental analysis and validating machine learning models.
</p>

<p style="text-align: justify;">
Probability and statistics are essential for understanding and developing machine learning algorithms. They provide the tools to model uncertainty, make inferences from data, and evaluate the performance of models. By grasping fundamental concepts like probability distributions, expectation, variance, and covariance, we can analyze and interpret data effectively.
</p>

<p style="text-align: justify;">
Bayes' theorem and probability density functions are crucial for updating beliefs and working with continuous random variables. Statistical methods enable us to perform tasks like classification, hypothesis testing, and parameter estimation, which are integral to machine learning.
</p>

<p style="text-align: justify;">
Implementing these concepts in Rust allows us to leverage the language's performance and safety features. The practical examples provided demonstrate how to build probabilistic models, apply Bayesian inference, and conduct statistical analyses using Rust. By combining rigorous mathematical theory with practical implementation, we can develop robust machine learning applications capable of handling real-world data complexities.
</p>

# 3.4. Linear Models and Least Squares
<p style="text-align: justify;">
Linear models are fundamental tools in both regression and classification tasks within machine learning. They serve as the basis for many complex algorithms and provide intuitive geometric interpretations that facilitate understanding and analysis. The ordinary least squares (OLS) method is a key technique used to estimate the parameters of linear models, minimizing the discrepancies between observed and predicted values.
</p>

<p style="text-align: justify;">
A linear model assumes a linear relationship between the input variables (features) and the output variable (target). In the context of regression, the goal is to predict a continuous output variable $y$ based on a vector of input features $\mathbf{x} = [x_1, x_2, \dots, x_n]^T$. The linear regression model is expressed as:
</p>

<p style="text-align: justify;">
$$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon = \mathbf{x}^T \boldsymbol{\beta} + \epsilon $$
</p>
<p style="text-align: justify;">
where: $\boldsymbol{\beta} = [\beta_0, \beta_1, \dots, \beta_n]^T$ are the model parameters (coefficients), $\epsilon$ is the error term accounting for the discrepancy between the observed and predicted values.
</p>

<p style="text-align: justify;">
For classification tasks, linear models can be used to separate data into different classes by finding a hyperplane that best divides the feature space. One common linear classification model is the logistic regression, which models the probability of a binary outcome:
</p>

<p style="text-align: justify;">
$$ P(y = 1 | \mathbf{x}) = \sigma(\mathbf{x}^T \boldsymbol{\beta}) = \frac{1}{1 + e^{-\mathbf{x}^T \boldsymbol{\beta}}} $$
</p>
<p style="text-align: justify;">
The Ordinary Least Squares method is a statistical technique used to estimate the parameters $\boldsymbol{\beta}$ of a linear regression model. OLS aims to minimize the sum of the squared differences between the observed target values $y_i$ and the predicted values $\hat{y}_i$:
</p>

<p style="text-align: justify;">
$$ \min_{\boldsymbol{\beta}} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 = \min_{\boldsymbol{\beta}} \sum_{i=1}^{m} (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 $$
</p>
<p style="text-align: justify;">
where $m$ is the number of observations.
</p>

<p style="text-align: justify;">
Geometrically, the linear regression model seeks to find the hyperplane in the feature space that best fits the data points. In a two-dimensional feature space, this hyperplane reduces to a line. The parameters $\boldsymbol{\beta}$ define the orientation and position of this hyperplane.
</p>

<p style="text-align: justify;">
For linear classification, the hyperplane serves as a decision boundary that separates different classes. Data points are classified based on which side of the hyperplane they fall on.
</p>

<p style="text-align: justify;">
Linear models rely on several key assumptions:
</p>

- <p style="text-align: justify;">Linearity: The relationship between the independent variables and the dependent variable is linear.</p>
- <p style="text-align: justify;">Independence: The residuals (errors) are independent of each other.</p>
- <p style="text-align: justify;">Homoscedasticity: The residuals have constant variance at every level of the independent variables.</p>
- <p style="text-align: justify;">Normality: The residuals are normally distributed (important for inference).</p>
<p style="text-align: justify;">
Violations of these assumptions can lead to biased or inefficient estimates.
</p>

<p style="text-align: justify;">
To find the parameter vector $\boldsymbol{\beta}$ that minimizes the cost function in OLS, we set the derivative of the cost function with respect to $\boldsymbol{\beta}$ to zero and solve for $\boldsymbol{\beta}$.
</p>

<p style="text-align: justify;">
Let $\mathbf{X}$ be the design matrix of size $m \times n$, where each row represents an observation $\mathbf{x}_i^T$, and $\mathbf{y}$ is the vector of target values.
</p>

<p style="text-align: justify;">
The cost function is:
</p>

<p style="text-align: justify;">
$$ J(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) $$
</p>
<p style="text-align: justify;">
Taking the derivative with respect to $\boldsymbol{\beta}$ and setting it to zero:
</p>

<p style="text-align: justify;">
$$ \frac{\partial J}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = 0 $$
</p>
<p style="text-align: justify;">
Solving for $\boldsymbol{\beta}$:
</p>

<p style="text-align: justify;">
$$ \mathbf{X}^T \mathbf{X} \boldsymbol{\beta} = \mathbf{X}^T \mathbf{y} $$
</p>
<p style="text-align: justify;">
Assuming $\mathbf{X}^T \mathbf{X}$ is invertible, the least squares solution is:
</p>

<p style="text-align: justify;">
$$ \boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} $$
</p>
<p style="text-align: justify;">
Linear models form the foundation of many machine learning algorithms due to their simplicity and interpretability. They are often used as a starting point before exploring more complex models. Techniques like support vector machines (SVMs), neural networks, and even deep learning architectures build upon the concepts introduced by linear models.
</p>

<p style="text-align: justify;">
We will implement linear regression using the OLS method in Rust. We will also introduce regularization techniques like Ridge (L2 regularization) and Lasso (L1 regularization) to prevent overfitting.
</p>

{{< prism lang="rust" line-numbers="true">}}
use ndarray::{array, Array2, Array1};
use ndarray_linalg::Inverse;

fn main() {
    // Example data as before
    let x_data = array![
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
        [1.0, 5.0],
    ];
    let y_data = array![2.0, 3.0, 4.0, 5.0, 6.0];

    let x: Array2<f64> = Array2::from(x_data);  // Specify f64 type
    let y: Array1<f64> = Array1::from(y_data);  // Specify f64 type

    // Regularization parameter
    let lambda = 1.0;

    // Compute beta = (X^T X + λI)^(-1) X^T y
    let xt = x.t();
    let xt_x = xt.dot(&x);
    let identity = Array2::<f64>::eye(xt_x.shape()[0]);  // Specify f64 type for identity matrix
    let xt_x_reg = &xt_x + &(identity * lambda);
    let xt_x_reg_inv = xt_x_reg.inv().expect("Matrix is singular and cannot be inverted.");
    let xt_y = xt.dot(&y);
    let beta = xt_x_reg_inv.dot(&xt_y);

    println!("Coefficients with Ridge Regression:\n{:?}", beta);
}
{{< /prism >}}
<p style="text-align: justify;">
In this code, we utilize the <code>ndarray</code> crate for numerical computations, which offers efficient array structures and operations suitable for handling multi-dimensional data in Rust. The feature matrix <code>x_data</code> includes a column of ones to account for the intercept term $β0\beta_0β0$, ensuring that the model can learn the bias term in the linear regression equation. This augmentation allows the linear model to fit data that does not necessarily pass through the origin.
</p>

<p style="text-align: justify;">
We compute the transpose of the feature matrix <code>x</code> and perform matrix multiplications to obtain the parameter vector $\boldsymbol{\beta}$. Specifically, we calculate $\mathbf{X}^T \mathbf{X}$ and $\mathbf{X}^T \mathbf{y}$, which are essential components in deriving the normal equation for the least squares solution. The multiplication of the transposed matrix with the original matrix and the target vector encapsulates the relationships between features and the target variable across all samples.
</p>

<p style="text-align: justify;">
The <code>inv()</code> function is employed to compute the inverse of the matrix $\mathbf{X}^T \mathbf{X}$. Calculating this inverse is a critical step in solving the normal equation $\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$, which provides the closed-form solution for the ordinary least squares estimation. We include error handling to address cases where the matrix $\mathbf{X}^T \mathbf{X}$ may be singular or nearly singular, making it non-invertible. If the matrix is singular, the program will output an appropriate error message, indicating that the inverse cannot be computed and that the least squares solution is undefined in this scenario.
</p>

<p style="text-align: justify;">
The resulting coefficients stored in the variable <code>beta</code> contain the estimated parameters of the linear model. These coefficients represent the weights assigned to each feature, including the intercept term, and define the hyperplane that best fits the data by minimizing the sum of the squared differences between the observed target values and the values predicted by the model. The computed $\boldsymbol{\beta}$ can then be used to make predictions on new data or to analyze the relationship between the features and the target variable.
</p>

<p style="text-align: justify;">
Ridge regression adds a penalty term to the cost function to prevent overfitting:
</p>

<p style="text-align: justify;">
$$J(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) + \lambda \boldsymbol{\beta}^T \boldsymbol{\beta}$$
</p>
<p style="text-align: justify;">
The solution becomes:
</p>

<p style="text-align: justify;">
$$ \boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y} $$
</p>
{{< prism lang="rust" line-numbers="true">}}
use ndarray::{array, Array1, Array2};
use ndarray_linalg::Inverse;

fn main() {
    // Example data as before
    let x_data = array![
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
        [1.0, 5.0],
    ];
    let y_data = array![2.0, 3.0, 4.0, 5.0, 6.0];

    // Explicitly specify f64 for numerical operations
    let x: Array2<f64> = Array2::from(x_data);  
    let y: Array1<f64> = Array1::from(y_data);  

    // Regularization parameter
    let lambda = 1.0;

    // Compute beta = (X^T X + λI)^(-1) X^T y
    let xt = x.t();
    let xt_x = xt.dot(&x);
    
    // Explicitly specify the element type for the identity matrix
    let identity: Array2<f64> = Array2::eye(xt_x.shape()[0]);
    let xt_x_reg = &xt_x + &(identity * lambda);
    let xt_x_reg_inv = xt_x_reg.inv().expect("Matrix is singular and cannot be inverted.");
    let xt_y = xt.dot(&y);
    let beta = xt_x_reg_inv.dot(&xt_y);

    println!("Coefficients with Ridge Regression:\n{:?}", beta);
}
{{< /prism >}}
<p style="text-align: justify;">
In this implementation, we introduce a regularization parameter $\lambda$ to control the amount of regularization applied to the linear regression model. By adjusting $\lambda$, we can influence the strength of the penalty imposed on the magnitude of the coefficients. The goal of this regularization is to prevent overfitting by discouraging the model from assigning excessively large weights to the features, which might capture noise rather than the underlying pattern in the data.
</p>

<p style="text-align: justify;">
To incorporate this regularization into the model, we modify the normal equation used in ordinary least squares regression. Specifically, we scale the identity matrix $\mathbf{I}$ by the regularization parameter $\lambda$ and add it to the matrix $\mathbf{X}^T \mathbf{X}$, resulting in the equation:
</p>

<p style="text-align: justify;">
$$ \boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y} $$
</p>
<p style="text-align: justify;">
This modification effectively adds a penalty term $\lambda \boldsymbol{\beta}^T \boldsymbol{\beta}$ to the cost function, which penalizes large coefficients. By doing so, it reduces the variance of the model without substantially increasing the bias, thus improving the model's ability to generalize to unseen data. This regularization technique is known as Ridge Regression or $L2$ regularization.
</p>

<p style="text-align: justify;">
The rest of the computation follows similarly to the ordinary least squares (OLS) solution. After adjusting $\mathbf{X}^T \mathbf{X}$ with the regularization term, we compute the inverse of this modified matrix. We then multiply the inverse by $\mathbf{X}^T \mathbf{y}$ to solve for the regularized coefficient vector $\boldsymbol{\beta}$. This solution provides us with coefficients that not only fit the training data but also maintain a balance between model complexity and predictive performance, leading to better generalization on new data.
</p>

<p style="text-align: justify;">
Lasso regression adds an $L1$ penalty to the cost function:
</p>

<p style="text-align: justify;">
$$ J(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) + \lambda \|\boldsymbol{\beta}\|_1 $$
</p>
<p style="text-align: justify;">
Unlike Ridge regression, Lasso does not have a closed-form solution, and iterative optimization methods are required. We can use coordinate descent to implement Lasso.
</p>

{{< prism lang="rust" line-numbers="true">}}
use ndarray::{array, Array1, Array2};

fn main() {
    // Example data as before
    let x_data = array![
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
        [1.0, 5.0],
    ];
    let y_data = array![2.0, 3.0, 4.0, 5.0, 6.0];

    let x = Array2::from(x_data);
    let y = Array1::from(y_data);

    // Regularization parameter
    let lambda = 0.1;
    let tol = 1e-4;
    let max_iter = 1000;

    // Initialize coefficients
    let mut beta = Array1::zeros(x.shape()[1]);

    for _ in 0..max_iter {
        let beta_old = beta.clone();

        for j in 0..beta.len() {
            // Compute residual
            let residual = &y - &x.dot(&beta);

            // Adjust residual to account for the current beta[j]
            let adjusted_residual = &residual + &x.column(j).mapv(|v| v * beta[j]);

            let rho = x.column(j).dot(&adjusted_residual);
            let x_j_norm = x.column(j).mapv(|v| f64::powi(v, 2)).sum();

            // Update beta[j] using the soft-thresholding function
            let beta_j = soft_threshold(rho / x_j_norm, lambda / x_j_norm);
            beta[j] = beta_j;
        }

        // Check for convergence
        if (&beta - &beta_old).mapv(f64::abs).sum() < tol {
            break;
        }
    }

    println!("Coefficients with Lasso Regression:\n{:?}", beta);
}

// Soft-thresholding function for Lasso
fn soft_threshold(value: f64, threshold: f64) -> f64 {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        0.0
    }
}
{{< /prism >}}
<p style="text-align: justify;">
In this implementation, we employ the coordinate descent algorithm to perform Lasso regression, which involves updating one coefficient at a time while holding the others fixed. Coordinate descent simplifies the optimization problem by breaking it down into a series of one-dimensional sub-problems. At each iteration, we cycle through the coefficients $\beta_j$ for $j = 1, 2, \dots, n$, where $n$ is the number of features. For each coefficient, we solve for the optimal value that minimizes the cost function while keeping the other coefficients constant. This approach is particularly efficient for high-dimensional data and models where the cost function is separable with respect to the parameters.
</p>

<p style="text-align: justify;">
The <code>soft_threshold</code> function is a critical component of this algorithm. It applies the proximal operator for the $L1$ norm, which is essential in enforcing the sparsity constraint characteristic of Lasso regression. The $L1$ regularization introduces a penalty proportional to the absolute value of the coefficients, promoting solutions where some coefficients are exactly zero. The soft thresholding operation adjusts the updated coefficient $\beta_j$ by shrinking it towards zero by an amount proportional to the regularization parameter $\lambda$. Mathematically, the soft thresholding function is defined as:
</p>

<p style="text-align: justify;">
$$ \text{soft\_threshold}(z, \gamma) = \begin{cases} z - \gamma, & \text{if } z > \gamma \\ 0, & \text{if } |z| \leq \gamma \\ z + \gamma, & \text{if } z < -\gamma \\ \end{cases} $$
</p>
<p style="text-align: justify;">
where $z$ is the coordinate-wise update and $\gamma = \lambda / x_j^T x_j$ is the threshold scaled by the feature's magnitude.
</p>

<p style="text-align: justify;">
We iterate this process until the change in the coefficients between successive iterations falls below a specified tolerance level or until a maximum number of iterations is reached. The convergence criterion ensures that the algorithm continues refining the coefficients only while significant improvements are being made. This iterative refinement allows the model to progressively approach the optimal set of parameters that minimize the cost function while adhering to the sparsity constraint imposed by the $L1$ regularization.
</p>

<p style="text-align: justify;">
This method is highly effective in handling high-dimensional data and performing feature selection. By shrinking some coefficients to exactly zero, the Lasso regression effectively excludes the corresponding features from the model. This not only reduces the complexity of the model but also enhances interpretability by identifying the most influential predictors. The ability to perform variable selection makes coordinate descent with $L1$ regularization particularly valuable in situations where the number of features is large, and only a subset is relevant to the target variable.
</p>

<p style="text-align: justify;">
The combination of coordinate descent and soft thresholding in Lasso regression provides a robust and efficient approach to linear modeling, especially in contexts where feature selection and model sparsity are desired outcomes. By carefully controlling the regularization parameter λ\\lambdaλ, we can balance the trade-off between model complexity and predictive accuracy, leading to models that generalize well to new, unseen data.
</p>

<p style="text-align: justify;">
To apply these models to real-world datasets, we need to handle data loading, preprocessing, and evaluation.
</p>

#### **Example:** Predicting House Prices
<p style="text-align: justify;">
Suppose we have a dataset with features like square footage, number of bedrooms, and age of the house, and we want to predict the house price.
</p>

{{< prism lang="rust" line-numbers="true">}}
use csv;
use ndarray::{Array2, Axis};
use ndarray_csv::Array2Reader;
use std::error::Error;
use std::fs::File;

fn main() -> Result<(), Box<dyn Error>> {
    // Load data from CSV
    let file = File::open("housing_data.csv")?;
    let mut reader = csv::Reader::from_reader(file);
    let array: Array2<f64> = reader.deserialize_array2_dynamic()?;

    // Assume the last column is the target variable
    let (x, y) = array.view().split_at(Axis(1), array.shape()[1] - 1);
    let x = x.to_owned();
    let y = y.column(0).to_owned();

    // Add intercept term
    let ones = Array2::ones((x.nrows(), 1));
    let x = ndarray::concatenate![Axis(1), ones, x];

    // Proceed with OLS or regularized regression as before
    // ...

    Ok(())
}
{{< /prism >}}
<p style="text-align: justify;">
In this example, we utilize the <code>ndarray_csv</code> crate to read CSV data into an <code>Array2<f64></code>, allowing us to efficiently load numerical data from a CSV file into an array suitable for computational tasks in Rust. After loading the data, we split the array into the feature matrix <code>x</code> and the target vector <code>y</code>, effectively separating the independent variables from the dependent variable we aim to predict. To include an intercept term in our linear regression model, we add a column of ones to the feature matrix <code>x</code>, enabling the model to learn the bias term alongside the other coefficients. With the data properly prepared, we can apply any of the regression methods implemented earlier—such as ordinary least squares, Ridge regression, or Lasso regression—to fit the model to the data and make predictions. Depending on the dataset, additional preprocessing steps like normalization to scale the features or handling missing values to ensure data integrity may be necessary to improve the model's performance and accuracy.
</p>

<p style="text-align: justify;">
For classification tasks, we can implement logistic regression using gradient descent.
</p>

{{< prism lang="rust" line-numbers="true">}}
use ndarray::{Array2, Array1, Axis};
use ndarray::array;

fn sigmoid(z: &Array1<f64>) -> Array1<f64> {
    z.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

fn main() {
    // Example data
    let x_data = array![
        [34.62, 78.02],
        [30.29, 43.89],
        [35.84, 72.90],
        [60.18, 86.31],
        [79.03, 75.34],
        [45.08, 56.32],
        [61.18, 96.51],
        [75.42, 69.12],
        [76.08, 86.64],
    ];
    let y_data = array![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];

    let x = Array2::from(x_data);
    let y = Array1::from(y_data);

    // Add intercept term
    let ones = Array2::ones((x.nrows(), 1));
    let x = ndarray::concatenate![Axis(1), ones, x];

    // Initialize parameters
    let mut beta = Array1::zeros(x.shape()[1]);
    let learning_rate = 0.001;
    let epochs = 10000;

    // Gradient descent
    for _ in 0..epochs {
        let z = x.dot(&beta);
        let predictions = sigmoid(&z);
        let errors = &y - &predictions;
        let gradient = x.t().dot(&errors) / x.nrows() as f64;
        beta += &(&gradient * learning_rate);
    }

    println!("Coefficients (beta):\n{:?}", beta);

    // Make predictions
    let test_point = array![1.0, 45.0, 85.0]; // Include intercept term
    let z_test = test_point.dot(&beta);
    let probability = sigmoid(&array![z_test]); // Wrap the result in an array
    println!("Predicted probability: {:?}", probability);
}
{{< /prism >}}
<p style="text-align: justify;">
In this implementation, we begin by defining the sigmoid function, which maps the linear predictions to probabilities between 0 and 1, enabling the model to output meaningful probabilities for classification tasks. To include the intercept term (bias) in our logistic regression model, we augment the feature matrix $x$ by adding a column of ones, allowing the model to learn the intercept alongside the weights for each feature. We initialize the parameter vector $\beta$ to zeros, providing a starting point for the gradient descent optimization. Gradient descent is then employed to minimize the logistic loss function (also known as the cross-entropy loss), iteratively updating the parameters to reduce the difference between the predicted probabilities and the actual class labels. In each iteration, we compute the predictions by applying the sigmoid function to the linear combination of features and parameters, calculate the errors by comparing these predictions to the true labels, compute the gradient of the loss with respect to the parameters, and update the parameters in the direction that minimizes the loss. After the training process converges, the model can be used to predict the probability that a new data point belongs to the positive class by feeding its features into the trained model, thus enabling probabilistic classification based on the learned relationships in the data.
</p>

<p style="text-align: justify;">
We can add regularization to logistic regression to prevent overfitting, similar to Ridge regression.
</p>

{{< prism lang="rust" line-numbers="true">}}
// Inside the gradient descent loop
let regularization_strength = 0.1;
for _ in 0..epochs {
    let z = x.dot(&beta);
    let predictions = sigmoid(&z);
    let errors = &y - &predictions;
    let gradient = x.t().dot(&errors) / x.nrows() as f64 - &(&beta * regularization_strength);
    beta += &(&gradient * learning_rate);
}
{{< /prism >}}
<p style="text-align: justify;">
In the regularized logistic regression implementation, we incorporate regularization by subtracting the regularization term from the gradient during each iteration of the gradient descent algorithm. The <code>regularization_strength</code> parameter controls the amount of regularization applied, effectively determining how much penalty is imposed on the magnitude of the coefficients. This process penalizes large coefficients by reducing their values, which discourages the model from fitting the noise in the training data and thus improves generalization to unseen data by preventing overfitting.
</p>

<p style="text-align: justify;">
Linear models and the least squares method are foundational concepts in machine learning, providing a starting point for understanding more complex algorithms. By implementing these models from scratch in Rust, we gain deeper insights into their workings and the underlying mathematics. Rust's performance and safety make it an excellent choice for numerical computations and machine learning applications.
</p>

<p style="text-align: justify;">
Through the practical examples, we have demonstrated how to:
</p>

- <p style="text-align: justify;">Implement ordinary least squares linear regression.</p>
- <p style="text-align: justify;">Incorporate regularization techniques like Ridge and Lasso regression.</p>
- <p style="text-align: justify;">Fit linear models to real-world data, handling data loading and preprocessing.</p>
- <p style="text-align: justify;">Implement linear classification models like logistic regression and add regularization.</p>
<p style="text-align: justify;">
By combining rigorous mathematical theory with hands-on coding experience, we build a solid foundation for developing and understanding machine learning models using Rust.
</p>

# 3.5. Numerical Methods
<p style="text-align: justify;">
Numerical methods are essential tools in machine learning and computational mathematics, enabling the approximation of solutions to problems that cannot be solved analytically. They play a crucial role in optimization problems where analytical solutions are infeasible due to the complexity or non-linearity of the functions involved. Numerical methods allow us to perform differentiation and integration numerically, and to find roots of equations, which are fundamental operations in training machine learning models.
</p>

<p style="text-align: justify;">
Differentiation involves computing the derivative of a function, representing the rate at which the function's value changes with respect to changes in its input. In many cases, especially in machine learning, the functions we deal with are complex and do not have derivatives that can be expressed in closed-form analytical expressions. Numerical differentiation provides a way to approximate the derivative using finite differences, making it possible to compute gradients needed for optimization algorithms.
</p>

<p style="text-align: justify;">
One common method is the finite difference approximation. For a function $f(x)$, the first derivative at a point $x$ can be approximated using the forward difference formula:
</p>

<p style="text-align: justify;">
$$f'(x) \approx \frac{f(x + h) - f(x)}{h}$$
</p>
<p style="text-align: justify;">
Alternatively, the central difference formula provides a more accurate approximation by considering points on both sides of $x$:
</p>

<p style="text-align: justify;">
$$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$
</p>
<p style="text-align: justify;">
Here, $h$ is a small step size. The choice of $h$ is critical; too large, and the approximation loses accuracy; too small, and numerical errors due to machine precision can become significant.
</p>

<p style="text-align: justify;">
Integration involves finding the area under the curve of a function, which is essential in calculating probabilities, expectations, and cumulative quantities in machine learning. When analytical integration is not possible, numerical integration methods such as the trapezoidal rule and Simpson's rule are employed to approximate definite integrals.
</p>

<p style="text-align: justify;">
The trapezoidal rule approximates the area under the curve by dividing it into trapezoids:
</p>

<p style="text-align: justify;">
$$ \int_a^b f(x) \, dx \approx \frac{h}{2} \left( f(a) + 2\sum_{k=1}^{n-1} f(a + kh) + f(b) \right) $$
</p>
<p style="text-align: justify;">
Where $h = \frac{b - a}{n}$ is the width of each subinterval, and nnn is the number of subintervals.
</p>

<p style="text-align: justify;">
Simpson's rule offers a higher-order approximation by fitting quadratic polynomials to the function segments:
</p>

<p style="text-align: justify;">
$$ \int_a^b f(x) \, dx \approx \frac{h}{3} \left( f(a) + 4\sum_{k=1}^{n/2} f(a + (2k - 1)h) + 2\sum_{k=1}^{n/2 - 1} f(a + 2kh) + f(b) \right) $$
</p>
<p style="text-align: justify;">
This method requires $n$ to be even and generally provides better accuracy than the trapezoidal rule for smooth functions.
</p>

<p style="text-align: justify;">
Root-finding algorithms are numerical methods used to find solutions to equations of the form $f(x) = 0$. These algorithms are fundamental in optimization, as finding the roots of the derivative of a function allows us to locate its maxima or minima.
</p>

<p style="text-align: justify;">
The Newton-Raphson method is an efficient iterative technique that uses the first derivative of the function:
</p>

<p style="text-align: justify;">
$$f'(x_n)x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$
</p>
<p style="text-align: justify;">
Starting from an initial guess $x_0$, the method refines the estimate of the root by considering the tangent at $x_n$ and finding where it crosses the $x$-axis.
</p>

<p style="text-align: justify;">
The bisection method is a robust root-finding algorithm that does not require derivatives. It works by repeatedly narrowing down an interval $[a, b]$ where the function changes sign (i.e., $f(a)f(b) < 0$). At each step, the interval is halved by selecting the midpoint $c = \frac{a + b}{2}$, and the subinterval where the sign change occurs is chosen for the next iteration. This method guarantees convergence but may be slower than Newton-Raphson.
</p>

<p style="text-align: justify;">
In machine learning, optimization problems are ubiquitous, whether in training neural networks, fitting models to data, or performing inference in probabilistic models. These problems often involve complex, high-dimensional functions that lack analytical solutions. Numerical methods become indispensable in such cases, providing practical means to approximate gradients, integrals, and solutions to equations that are critical for optimization algorithms.
</p>

<p style="text-align: justify;">
For instance, gradient-based optimization algorithms like gradient descent require the computation of gradients of the loss function with respect to model parameters. When the loss function is complex or derived from empirical data, analytical gradients may not be available or practical to compute. Numerical differentiation allows us to approximate these gradients efficiently.
</p>

<p style="text-align: justify;">
Similarly, numerical integration is vital in probabilistic machine learning, where expectations and marginalizations over probability distributions are required. When dealing with continuous random variables and complex likelihood functions, numerical integration techniques enable us to compute these quantities.
</p>

<p style="text-align: justify;">
Root-finding algorithms are essential for solving equations that arise in optimization, such as setting the derivative of the loss function to zero to find minima. They provide iterative procedures to find solutions when analytical methods are not feasible.
</p>

<p style="text-align: justify;">
Rust, with its emphasis on performance and safety, is well-suited for implementing numerical algorithms. Its strong typing system and memory safety guarantees help prevent common programming errors that can lead to incorrect results or crashes.
</p>

#### **Example:** Numerical Differentiation
<p style="text-align: justify;">
We can implement numerical differentiation using the central difference method in Rust.
</p>

{{< prism lang="rust" line-numbers="true">}}
fn numerical_derivative<F>(f: F, x: f64, h: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn main() {
    // Define the function f(x) = x^3
    let f = |x: f64| x.powi(3);

    let x = 2.0;
    let h = 1e-5;

    let derivative = numerical_derivative(f, x, h);

    println!("The numerical derivative of f at x = {} is {:.6}", x, derivative);
}
{{< /prism >}}
<p style="text-align: justify;">
In this code, we define a generic function <code>numerical_derivative</code> that takes a function <code>f</code>, a point <code>x</code>, and a small step size <code>h</code>. It computes the approximate derivative using the central difference formula. We then define a specific function $f(x) = x^3$ and compute its derivative at $x = 2.0$. The exact derivative at this point is $f'(2) = 12.0$, allowing us to compare the numerical approximation to the analytical result.
</p>

#### **Example:** Numerical Integration
<p style="text-align: justify;">
Implementing the trapezoidal rule for numerical integration in Rust:
</p>

{{< prism lang="rust" line-numbers="true">}}
fn trapezoidal_rule<F>(f: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));

    for i in 1..n {
        let x = a + i as f64 * h;
        sum += f(x);
    }

    sum * h
}

fn main() {
    // Integrate f(x) = sin(x) from 0 to pi
    let f = |x: f64| x.sin();

    let a = 0.0;
    let b = std::f64::consts::PI;
    let n = 1000;

    let integral = trapezoidal_rule(f, a, b, n);

    println!("The approximate integral of sin(x) from {} to {} is {:.6}", a, b, integral);
}
{{< /prism >}}
<p style="text-align: justify;">
Here, the <code>trapezoidal_rule</code> function approximates the integral of a function <code>f</code> over the interval $[a, b]$ by dividing it into <code>n</code> subintervals. We apply this function to compute the integral of $\sin(x)$ from $0$ to $\pi$. The exact value of this integral is $2$, and with $n = 1000$, the numerical approximation should be very close to the exact value, demonstrating the effectiveness of the method.
</p>

#### **Example:** Root-Finding with Newton-Raphson Method
<p style="text-align: justify;">
Implementing the Newton-Raphson method in Rust:
</p>

{{< prism lang="rust" line-numbers="true">}}
fn newton_raphson<F, G>(f: F, df: G, x0: f64, tol: f64, max_iter: usize) -> Option<f64>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    let mut x = x0;

    for _ in 0..max_iter {
        let fx = f(x);
        let dfx = df(x);

        if dfx.abs() < tol {
            // Derivative too small; avoid division by zero
            return None;
        }

        let x_new = x - fx / dfx;

        if (x_new - x).abs() < tol {
            return Some(x_new);
        }

        x = x_new;
    }

    None
}

fn main() {
    // Find the root of f(x) = x^2 - 2 (i.e., sqrt(2))
    let f = |x: f64| x.powi(2) - 2.0;
    let df = |x: f64| 2.0 * x;

    let x0 = 1.0;
    let tol = 1e-10;
    let max_iter = 100;

    match newton_raphson(f, df, x0, tol, max_iter) {
        Some(root) => println!("Root found: {:.10}", root),
        None => println!("Failed to find root."),
    }
}
{{< /prism >}}
<p style="text-align: justify;">
The <code>newton_raphson</code> function iteratively applies the Newton-Raphson update to find a root of the function <code>f</code>, given its derivative <code>df</code>. Starting from an initial guess <code>x0</code>, it updates the estimate until convergence within a specified tolerance or until the maximum number of iterations is reached. In this example, we find the square root of 2 by solving $x^2 - 2 = 0$.
</p>

<p style="text-align: justify;">
Numerical methods can be directly applied to optimize machine learning models, especially when dealing with complex loss functions or when analytical solutions are unavailable.
</p>

#### **Example:** Minimizing a Loss Function Using Numerical Gradient Descent
<p style="text-align: justify;">
Consider a loss function that is difficult to differentiate analytically:
</p>

{{< prism lang="rust" line-numbers="true">}}
fn numerical_gradient<F>(f: F, x: f64, h: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn gradient_descent<F>(f: F, x0: f64, learning_rate: f64, tol: f64, max_iter: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut x = x0;

    for _ in 0..max_iter {
        let grad = numerical_gradient(&f, x, 1e-5);
        let x_new = x - learning_rate * grad;

        if (x_new - x).abs() < tol {
            break;
        }

        x = x_new;
    }

    x
}

fn main() {
    // Define a complex loss function
    let f = |x: f64| (x - 3.0).powi(4) + (x - 3.0).powi(2) + x.sin();

    let x0 = 0.0;
    let learning_rate = 0.01;
    let tol = 1e-6;
    let max_iter = 10000;

    let min_x = gradient_descent(f, x0, learning_rate, tol, max_iter);

    println!("The minimum of f(x) occurs at x = {:.6}", min_x);
    println!("Minimum value of f(x) = {:.6}", f(min_x));
}
{{< /prism >}}
<p style="text-align: justify;">
We implement <code>numerical_gradient</code> to approximate the gradient of the loss function <code>f</code> at each iteration. The <code>gradient_descent</code> function updates the parameter <code>x</code> by moving in the opposite direction of the gradient, scaled by the learning rate. The algorithm continues until the change in <code>x</code> is less than the tolerance or until the maximum number of iterations is reached. This approach allows us to minimize complex loss functions numerically.
</p>

#### **Example:** Using Root-Finding to Solve for Optimal Parameters
<p style="text-align: justify;">
In some optimization problems, we can set the derivative of the loss function to zero and use root-finding methods to find the optimal parameters.
</p>

{{< prism lang="rust" line-numbers="true">}}
fn newton_optimize<F, G>(f_prime: F, f_double_prime: G, x0: f64, tol: f64, max_iter: usize) -> Option<f64>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    let mut x = x0;

    for _ in 0..max_iter {
        let fp = f_prime(x);
        let fpp = f_double_prime(x);

        if fpp.abs() < tol {
            // Second derivative too small; cannot proceed
            return None;
        }

        let x_new = x - fp / fpp;

        if (x_new - x).abs() < tol {
            return Some(x_new);
        }

        x = x_new;
    }

    None
}

fn main() {
    // Optimize f(x) = x^3 - 3x^2 + 2
    let f_prime = |x: f64| 3.0 * x.powi(2) - 6.0 * x;
    let f_double_prime = |x: f64| 6.0 * x - 6.0;

    let x0 = 0.0;
    let tol = 1e-6;
    let max_iter = 100;

    match newton_optimize(f_prime, f_double_prime, x0, tol, max_iter) {
        Some(opt_x) => {
            let min_f = opt_x.powi(3) - 3.0 * opt_x.powi(2) + 2.0;
            println!("Minimum of f(x) occurs at x = {:.6}", opt_x);
            println!("Minimum value of f(x) = {:.6}", min_f);
        }
        None => println!("Failed to find minimum."),
    }
}
{{< /prism >}}
<p style="text-align: justify;">
Here, we use the Newton-Raphson method to find the critical points of the function $f(x) = x^3 - 3x^2 + 2$ by solving $f'(x) = 0$. By providing both the first and second derivatives, we can efficiently find the minima or maxima of the function.
</p>

<p style="text-align: justify;">
Numerical methods are indispensable in machine learning for solving optimization problems that lack analytical solutions. They enable us to approximate derivatives, integrals, and roots of functions, which are fundamental operations in model training and inference. Implementing these methods in Rust allows us to leverage the language's performance and safety features, ensuring efficient and reliable computations.
</p>

<p style="text-align: justify;">
Through the practical examples provided, we have demonstrated how to implement numerical differentiation, integration, and root-finding algorithms in Rust. We also showcased how these methods can be applied to optimize machine learning models, emphasizing their importance in handling complex functions encountered in real-world applications.
</p>

<p style="text-align: justify;">
By combining rigorous mathematical foundations with practical implementation, we equip ourselves with the tools necessary to develop robust and efficient machine learning models using Rust. This integration of theory and practice is essential for advancing in the field of machine learning and for tackling increasingly complex computational challenges.
</p>

# 3.6. Discrete Mathematics and Graph Theory
<p style="text-align: justify;">
Discrete mathematics and graph theory form the backbone of various machine learning algorithms and data structures. They provide the tools to model complex relationships, optimize computations, and represent data in ways that facilitate efficient processing and analysis. Understanding these mathematical concepts is crucial for developing advanced machine learning models that can handle intricate data structures and perform sophisticated computations.
</p>

<p style="text-align: justify;">
Combinatorics is the branch of mathematics dealing with counting, arrangement, and combination of discrete structures. It provides the foundational techniques for analyzing the complexity of algorithms, optimizing computations, and understanding the possibilities within a finite system. In machine learning, combinatorics is essential for tasks like feature selection, where one might need to evaluate different combinations of features to determine the most informative subset.
</p>

<p style="text-align: justify;">
Graph theory, a significant area within discrete mathematics, studies graphs as mathematical structures used to model pairwise relations between objects. A graph $G = (V, E)$ consists of a set of vertices $V$ and a set of edges $E$ connecting pairs of vertices. Graphs can be directed or undirected, weighted or unweighted, and they can represent a wide range of real-world problems.
</p>

- <p style="text-align: justify;">Vertices (Nodes): Represent entities or data points.</p>
- <p style="text-align: justify;">Edges (Links): Represent relationships or interactions between entities.</p>
<p style="text-align: justify;">
Trees are a special type of graph with no cycles, and they play a critical role in machine learning algorithms like decision trees and random forests. A tree is a connected acyclic graph, meaning there is exactly one path between any two vertices.
</p>

<p style="text-align: justify;">
Paths and cycles are fundamental concepts in graph theory that are essential for understanding the structure and traversal of graphs. A path in a graph is defined as a sequence of edges that connects a sequence of distinct vertices, where each adjacent pair of vertices in the sequence is connected by an edge. Formally, a path $P$ can be represented as $P = (v_1, e_1, v_2, e_2, \dots, e_{n-1}, v_n)$, where $v_i$ are vertices and eie_iei are edges such that each edge $e_i$ connects vertices $v_i$ and $v_{i+1}$. Paths represent a way to navigate through a graph from one vertex to another without revisiting any vertex, capturing the notion of movement or connectivity within the network.
</p>

<p style="text-align: justify;">
A cycle is a special type of path that starts and ends at the same vertex, forming a closed loop. In a cycle, all the edges and intermediate vertices are distinct, except for the starting and ending vertex, which are the same. Formally, a cycle $C$ is a path where $v_1 = v_n$ and all $v_i$ for $1 < i < n$ are distinct. Cycles are significant because they indicate the presence of loops within the graph, which can affect properties like connectivity and traversability.
</p>

<p style="text-align: justify;">
Understanding paths and cycles is vital for algorithms that traverse or search graphs, such as depth-first search (DFS) and breadth-first search (BFS). These algorithms explore the graph by systematically visiting vertices and edges. DFS explores as far along each branch as possible before backtracking, effectively following paths to their deepest extent. It is particularly useful for detecting cycles and solving problems like topological sorting and finding connected components. BFS, on the other hand, explores all neighbors of a vertex before moving to the next level, which is essential for finding the shortest path in unweighted graphs and for level-order traversal.
</p>

<p style="text-align: justify;">
Detecting cycles is crucial in many applications. For example, in dependency resolution systems (like package managers), cycles can lead to deadlocks or inconsistencies. In scheduling tasks, cycles may indicate circular dependencies that prevent tasks from being completed. Therefore, algorithms that can identify and handle paths and cycles are fundamental in both theoretical and practical aspects of graph analysis.
</p>

<p style="text-align: justify;">
Graph theory concepts like paths, cycles, and trees have significant applications in machine learning, where they are used to model data structures, represent dependencies, and design algorithms that leverage the inherent structure of data.
</p>

<p style="text-align: justify;">
Decision Trees are a prime example of utilizing tree structures in machine learning. A decision tree is a flowchart-like model where each internal node represents a test on a feature (e.g., whether a variable exceeds a certain threshold), each branch corresponds to the outcome of that test, and each leaf node represents a class label (in classification) or a value (in regression). The paths from the root to the leaves represent sequences of decisions leading to a prediction. Decision trees are built by recursively partitioning the data based on feature values to maximize some criterion like information gain or Gini impurity reduction. They are intuitive and interpretable models, making them valuable for understanding how predictions are made.
</p>

<p style="text-align: justify;">
For example, in a decision tree for classifying whether an email is spam or not, an internal node might test whether the email contains the word "free." Depending on the outcome (yes or no), the path proceeds to other tests or to a leaf node that makes the final classification. The absence of cycles in the tree ensures that the decision-making process is straightforward and that each path from the root to a leaf represents a unique sequence of feature evaluations.
</p>

<p style="text-align: justify;">
Bayesian Networks, also known as belief networks, are probabilistic graphical models that represent a set of variables and their conditional dependencies via a directed acyclic graph (DAG). In a Bayesian network, nodes represent random variables, and edges represent direct probabilistic dependencies between these variables. The direction of the edges indicates the direction of influence. The acyclic nature of the graph (absence of cycles) ensures that the joint probability distribution over all variables can be decomposed into a product of conditional probabilities, facilitating efficient computation and inference.
</p>

<p style="text-align: justify;">
Bayesian networks are powerful tools for reasoning under uncertainty and are widely used in fields like diagnostics, bioinformatics, and natural language processing. They allow for the incorporation of prior knowledge and can handle incomplete data. For instance, in medical diagnosis, a Bayesian network can model the probabilistic relationships between diseases and symptoms, enabling the computation of the likelihood of various diseases given observed symptoms.
</p>

<p style="text-align: justify;">
Markov Random Fields (MRFs) are undirected graphical models that represent the conditional dependencies between random variables. In an MRF, nodes correspond to random variables, and edges indicate that two variables are conditionally dependent given all other variables. Unlike Bayesian networks, MRFs use undirected edges, capturing the notion of mutual or symmetric relationships without implying a directional influence.
</p>

<p style="text-align: justify;">
MRFs are extensively used in computer vision, image processing, and spatial statistics, where the spatial relationships between variables are important. For example, in image segmentation, each pixel can be considered a node, and edges connect neighboring pixels. The MRF models the probability that adjacent pixels belong to the same segment, allowing for the incorporation of spatial smoothness constraints in the segmentation process.
</p>

<p style="text-align: justify;">
These applications illustrate how graph theory provides a mathematical framework for representing complex dependencies and interactions in machine learning models. By leveraging graphs, machine learning algorithms can capture both the global structure and local relationships within data, leading to more accurate and robust models. Understanding paths and cycles within these graphs is essential for ensuring that the models are well-defined (e.g., avoiding cycles in Bayesian networks) and for designing efficient algorithms for learning and inference.
</p>

<p style="text-align: justify;">
Graph theory provides a natural way to model complex relationships and interactions within data. In many machine learning applications, data points are not independent but are connected through various types of relationships. Graphs can capture these connections, allowing algorithms to leverage the structure within the data.
</p>

<p style="text-align: justify;">
For example, in social network analysis, nodes represent individuals, and edges represent friendships or interactions. Machine learning algorithms can utilize this graph structure to predict user behavior, recommend connections, or detect communities.
</p>

<p style="text-align: justify;">
In recommendation systems, bipartite graphs can model users and items, with edges representing user-item interactions like ratings or purchases. Algorithms can then analyze these graphs to make personalized recommendations.
</p>

<p style="text-align: justify;">
Combinatorial optimization involves finding an optimal object from a finite set of objects. Many machine learning problems can be framed as combinatorial optimization tasks, such as:
</p>

- <p style="text-align: justify;">Feature Selection: Selecting the best subset of features from a large set to improve model performance.</p>
- <p style="text-align: justify;">Clustering: Partitioning data points into groups (clusters) where intra-cluster similarity is maximized, and inter-cluster similarity is minimized.</p>
- <p style="text-align: justify;">Graph Partitioning: Dividing a graph into parts while minimizing the number of edges between parts, useful in parallel computing and network analysis.</p>
<p style="text-align: justify;">
Graph algorithms like Dijkstra’s algorithm, Kruskal’s algorithm, and Prim’s algorithm are fundamental tools for solving such optimization problems. They help in finding shortest paths, minimum spanning trees, and optimal network structures, which are essential in various machine learning tasks.
</p>

<p style="text-align: justify;">
Rust, with its strong emphasis on performance and safety, is an excellent language for implementing complex graph algorithms. Its ownership model and memory safety guarantees help prevent common bugs like null pointer dereferences and data races, which are crucial when dealing with intricate data structures like graphs.
</p>

<p style="text-align: justify;">
Dijkstra's algorithm computes the shortest paths from a single source node to all other nodes in a weighted graph with non-negative edge weights.
</p>

{{< prism lang="rust" line-numbers="true">}}
use std::collections::BinaryHeap;
use std::cmp::Ordering;

#[derive(Copy, Clone, Eq, PartialEq)]
struct State {
    cost: u64,
    position: usize,
}

// Implement ordering for the priority queue (min-heap)
impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Invert the ordering to make BinaryHeap act as min-heap
        other.cost.cmp(&self.cost)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &State) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn dijkstra(adj_list: &Vec<Vec<(usize, u64)>>, start: usize) -> Vec<u64> {
    let mut dist = vec![u64::MAX; adj_list.len()];
    let mut heap = BinaryHeap::new();

    dist[start] = 0;
    heap.push(State { cost: 0, position: start });

    while let Some(State { cost, position }) = heap.pop() {
        // Skip if we have already found a better path
        if cost > dist[position] {
            continue;
        }

        // Examine neighbors
        for &(neighbor, weight) in &adj_list[position] {
            let next = State {
                cost: cost + weight,
                position: neighbor,
            };

            // If shorter path to neighbor is found
            if next.cost < dist[next.position] {
                dist[next.position] = next.cost;
                heap.push(next);
            }
        }
    }

    dist
}

fn main() {
    // Example graph represented as an adjacency list
    // Each node has a list of (neighbor_index, weight) pairs
    let adj_list = vec![
        vec![(1, 2), (2, 5)],       // Node 0
        vec![(0, 2), (2, 3), (3, 1)], // Node 1
        vec![(0, 5), (1, 3), (3, 2)], // Node 2
        vec![(1, 1), (2, 2)],       // Node 3
    ];

    let start_node = 0;
    let distances = dijkstra(&adj_list, start_node);

    for (i, &d) in distances.iter().enumerate() {
        println!("Distance from node {} to node {} is {}", start_node, i, d);
    }
}
{{< /prism >}}
<p style="text-align: justify;">
In this Rust implementation of Dijkstra's algorithm, the graph is represented as an adjacency list <code>adj_list</code>, where each index corresponds to a node, and each element is a vector of tuples representing the neighboring nodes along with the weights of the edges connecting them. A <code>State</code> struct is defined to encapsulate the current position (node index) and the accumulated cost (distance from the start node). To enable the <code>BinaryHeap</code> to function as a min-heap—prioritizing states with lower costs—the <code>Ord</code> and <code>PartialOrd</code> traits are implemented for the <code>State</code> struct by reversing the ordering. The <code>dijkstra</code> function initializes a distance vector <code>dist</code> with maximum values (<code>u64::MAX</code>) to represent the initial unknown distances to each node and sets the distance to the start node as zero. A binary heap (priority queue) is used to efficiently select the next node with the smallest tentative distance, ensuring that the algorithm always explores the most promising node next. In the main loop, the state with the lowest cost is popped from the heap, and the algorithm updates the distances to its neighboring nodes if a shorter path is found through the current node. After executing the algorithm, the <code>dist</code> vector contains the shortest distances from the start node to all other nodes in the graph, effectively solving the single-source shortest path problem.
</p>

<p style="text-align: justify;">
A Minimum Spanning Tree (MST) connects all the nodes in a graph with the minimal total edge weight without any cycles. Kruskal's algorithm is a greedy method for finding an MST.
</p>

{{< prism lang="rust" line-numbers="true">}}
use std::cmp::Ordering;

#[derive(Clone)]
struct Edge {
    node1: usize,
    node2: usize,
    weight: u64,
}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other.weight.cmp(&self.weight)
    }
}

impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Edge) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Edge {
    fn eq(&self, other: &Edge) -> bool {
        self.weight == other.weight
    }
}

impl Eq for Edge {}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        UnionFind {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, u: usize) -> usize {
        if self.parent[u] != u {
            self.parent[u] = self.find(self.parent[u]); // Path compression
        }
        self.parent[u]
    }

    fn union(&mut self, u: usize, v: usize) {
        let u_root = self.find(u);
        let v_root = self.find(v);

        if u_root == v_root {
            return;
        }

        // Union by rank
        if self.rank[u_root] < self.rank[v_root] {
            self.parent[u_root] = v_root;
        } else {
            self.parent[v_root] = u_root;
            if self.rank[u_root] == self.rank[v_root] {
                self.rank[u_root] += 1;
            }
        }
    }
}

fn kruskal_mst(nodes: usize, edges: &mut Vec<Edge>) -> Vec<Edge> {
    // Sort edges by weight
    edges.sort();

    let mut uf = UnionFind::new(nodes);
    let mut mst = Vec::new();

    for edge in edges {
        if uf.find(edge.node1) != uf.find(edge.node2) {
            uf.union(edge.node1, edge.node2);
            mst.push(edge.clone());
        }
    }

    mst
}

fn main() {
    // Define edges of the graph
    let mut edges = vec![
        Edge { node1: 0, node2: 1, weight: 4 },
        Edge { node1: 0, node2: 2, weight: 4 },
        Edge { node1: 1, node2: 2, weight: 2 },
        Edge { node1: 1, node2: 3, weight: 5 },
        Edge { node1: 2, node2: 3, weight: 5 },
        Edge { node1: 2, node2: 4, weight: 9 },
        Edge { node1: 3, node2: 4, weight: 4 },
    ];

    let nodes = 5;
    let mst = kruskal_mst(nodes, &mut edges);

    println!("Edges in the Minimum Spanning Tree:");
    for edge in mst {
        println!("Edge from {} to {} with weight {}", edge.node1, edge.node2, edge.weight);
    }
}
{{< /prism >}}
<p style="text-align: justify;">
In this Rust implementation of Kruskal's algorithm, we start by defining an <code>Edge</code> struct to represent the edges between nodes, each with a specified weight, effectively modeling the connections in the graph. To facilitate the sorting of edges based on their weights, we implement the <code>Ord</code> and <code>PartialOrd</code> traits for the <code>Edge</code> struct, allowing the edges to be ordered from the smallest to the largest weight. A crucial component of the algorithm is the <code>UnionFind</code> data structure (also known as Disjoint Set Union), which efficiently keeps track of the connected components in the graph and detects cycles by supporting <code>find</code> and <code>union</code> operations. These operations are optimized using path compression and union by rank techniques, significantly improving performance when managing large graphs. The <code>kruskal_mst</code> function embodies the core logic of Kruskal's algorithm: it first sorts all the edges by weight and then iteratively adds them to the Minimum Spanning Tree (MST) if they do not form a cycle, which is determined by checking if the nodes connected by an edge are already in the same set within the <code>UnionFind</code> structure. In the <code>main</code> function, we define the edges of the graph and invoke the <code>kruskal_mst</code> function to compute the MST. After executing the algorithm, the resulting edges included in the MST are printed out, illustrating the minimal connections required to span all nodes while ensuring that the total weight is minimized and no cycles are formed.
</p>

<p style="text-align: justify;">
Decision trees are powerful machine learning models that recursively partition the feature space to make predictions. Implementing a decision tree using Rust language involves selecting the best feature and threshold at each node to split the data, aiming to maximize information gain or reduce impurity.
</p>

{{< prism lang="rust" line-numbers="true">}}
use std::collections::HashMap;

#[derive(Debug)]
struct DecisionTreeNode {
    feature_index: Option<usize>,
    threshold: Option<f64>,
    left: Option<Box<DecisionTreeNode>>,
    right: Option<Box<DecisionTreeNode>>,
    predicted_class: Option<usize>,
}

impl DecisionTreeNode {
    fn new_leaf(class: usize) -> Self {
        DecisionTreeNode {
            feature_index: None,
            threshold: None,
            left: None,
            right: None,
            predicted_class: Some(class),
        }
    }

    fn new_node(feature_index: usize, threshold: f64, left: DecisionTreeNode, right: DecisionTreeNode) -> Self {
        DecisionTreeNode {
            feature_index: Some(feature_index),
            threshold: Some(threshold),
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            predicted_class: None,
        }
    }
}

struct DecisionTreeClassifier {
    root: DecisionTreeNode,
    max_depth: usize,
    min_samples_split: usize,
}

impl DecisionTreeClassifier {
    fn new(max_depth: usize, min_samples_split: usize) -> Self {
        DecisionTreeClassifier {
            root: DecisionTreeNode {
                feature_index: None,
                threshold: None,
                left: None,
                right: None,
                predicted_class: None,
            },
            max_depth,
            min_samples_split,
        }
    }

    fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<usize>) {
        self.root = self.build_tree(x, y, 0);
    }

    fn build_tree(&self, x: &Vec<Vec<f64>>, y: &Vec<usize>, depth: usize) -> DecisionTreeNode {
        if y.len() < self.min_samples_split || depth >= self.max_depth {
            let class = self.majority_class(y);
            return DecisionTreeNode::new_leaf(class);
        }

        let (best_feature, best_threshold, best_gain) = self.best_split(x, y);
        if best_gain == 0.0 {
            let class = self.majority_class(y);
            return DecisionTreeNode::new_leaf(class);
        }

        let (left_indices, right_indices) = self.partition(x, best_feature, best_threshold);

        let left = self.build_tree(
            &left_indices.iter().map(|&i| x[i].clone()).collect(),
            &left_indices.iter().map(|&i| y[i]).collect(),
            depth + 1,
        );
        let right = self.build_tree(
            &right_indices.iter().map(|&i| x[i].clone()).collect(),
            &right_indices.iter().map(|&i| y[i]).collect(),
            depth + 1,
        );

        DecisionTreeNode::new_node(best_feature, best_threshold, left, right)
    }

    fn majority_class(&self, y: &Vec<usize>) -> usize {
        let mut counts = HashMap::new();
        for &label in y {
            *counts.entry(label).or_insert(0) += 1;
        }
        *counts.iter().max_by_key(|&(_, count)| count).unwrap().0
    }

    fn best_split(&self, x: &Vec<Vec<f64>>, y: &Vec<usize>) -> (usize, f64, f64) {
        let num_features = x[0].len();
        let mut best_gain = 0.0;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;

        let current_impurity = self.gini(y);

        for feature_index in 0..num_features {
            let mut thresholds: Vec<f64> = x.iter().map(|row| row[feature_index]).collect();
            thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());

            for &threshold in &thresholds {
                let (left_indices, right_indices) = self.partition(x, feature_index, threshold);

                if left_indices.is_empty() || right_indices.is_empty() {
                    continue;
                }

                let left_labels: Vec<usize> = left_indices.iter().map(|&i| y[i]).collect();
                let right_labels: Vec<usize> = right_indices.iter().map(|&i| y[i]).collect();

                let left_impurity = self.gini(&left_labels);
                let right_impurity = self.gini(&right_labels);

                let left_weight = left_labels.len() as f64 / y.len() as f64;
                let right_weight = right_labels.len() as f64 / y.len() as f64;

                let gain = current_impurity - (left_weight * left_impurity + right_weight * right_impurity);

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feature_index;
                    best_threshold = threshold;
                }
            }
        }

        (best_feature, best_threshold, best_gain)
    }

    fn partition(&self, x: &Vec<Vec<f64>>, feature_index: usize, threshold: f64) -> (Vec<usize>, Vec<usize>) {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for (i, row) in x.iter().enumerate() {
            if row[feature_index] <= threshold {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }

        (left_indices, right_indices)
    }

    fn gini(&self, y: &Vec<usize>) -> f64 {
        let mut counts = HashMap::new();
        for &label in y {
            *counts.entry(label).or_insert(0) += 1;
        }

        let total = y.len() as f64;
        let mut impurity = 1.0;

        for &count in counts.values() {
            let prob = count as f64 / total;
            impurity -= prob.powi(2);
        }

        impurity
    }

    fn predict(&self, x: &Vec<f64>) -> usize {
        let mut node = &self.root;
        while node.predicted_class.is_none() {
            let feature_index = node.feature_index.unwrap();
            let threshold = node.threshold.unwrap();

            if x[feature_index] <= threshold {
                node = node.left.as_ref().unwrap();
            } else {
                node = node.right.as_ref().unwrap();
            }
        }

        node.predicted_class.unwrap()
    }
}

fn main() {
    // Sample dataset (features and labels)
    let x = vec![
        vec![2.771244718, 1.784783929],
        vec![1.728571309, 1.169761413],
        vec![3.678319846, 2.81281357],
        vec![3.961043357, 2.61995032],
        vec![2.999208922, 2.209014212],
        vec![7.497545867, 3.162953546],
        vec![9.00220326, 3.339047188],
        vec![7.444542326, 0.476683375],
        vec![10.12493903, 3.234550982],
        vec![6.642287351, 3.319983761],
    ];
    let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

    let mut clf = DecisionTreeClassifier::new(3, 2);
    clf.fit(&x, &y);

    let test_sample = vec![3.0, 1.5];
    let prediction = clf.predict(&test_sample);

    println!("Predicted class for {:?} is {}", test_sample, prediction);
}
{{< /prism >}}
<p style="text-align: justify;">
Graph theory concepts can be applied to various machine learning tasks beyond traditional graph algorithms. For instance:
</p>

- <p style="text-align: justify;">Graph-Based Semi-Supervised Learning: Utilize graph structures to propagate labels from labeled to unlabeled data points based on their connectivity.</p>
- <p style="text-align: justify;">Spectral Clustering: Use the eigenvectors of the graph Laplacian matrix derived from the data to perform clustering.</p>
- <p style="text-align: justify;">Graph Neural Networks (GNNs): Extend neural networks to work directly on graph structures, allowing for learning representations of nodes, edges, and entire graphs.</p>
<p style="text-align: justify;">
Implementing such advanced models requires a deep understanding of graph theory and the ability to represent and manipulate graphs efficiently in code. Rust's performance and memory safety make it a suitable choice for developing high-performance machine learning libraries and applications that leverage these concepts.
</p>

<p style="text-align: justify;">
Discrete mathematics and graph theory are fundamental to many aspects of machine learning, providing the mathematical structures and algorithms needed to model complex relationships and optimize computations. By understanding graphs, trees, paths, and cycles, we can design algorithms that efficiently process and analyze data with intricate interconnections.
</p>

<p style="text-align: justify;">
Implementing graph algorithms in Rust allows us to harness the language's strengths in safety and performance. Through the examples of Dijkstra's algorithm, Kruskal's algorithm, and decision tree implementation, we have demonstrated how to apply these mathematical concepts practically. These implementations not only solidify our understanding of the underlying mathematics but also provide a foundation for building more complex machine learning models that can handle structured data.
</p>

<p style="text-align: justify;">
By integrating graph theory concepts into machine learning tasks, we can develop models that are better suited for data with inherent relational structures, such as social networks, molecular structures, and knowledge graphs. This integration is crucial for advancing machine learning applications in fields where relationships and interactions are as important as the individual data points themselves.
</p>

<p style="text-align: justify;">
In conclusion, mastering discrete mathematics and graph theory equips us with powerful tools to tackle a wide range of machine learning challenges, and implementing these concepts in Rust ensures that our solutions are robust, efficient, and reliable.
</p>

# 3.7. Conclusion
<p style="text-align: justify;">
By the end of Chapter 3, you will have a strong grasp of the mathematical principles that power machine learning algorithms and the ability to implement these concepts in Rust. This foundation is crucial for understanding more advanced topics in machine learning and for developing effective, optimized models.
</p>

## 3.7.1. Further Learning with GenAI
<p style="text-align: justify;">
To deepen your understanding of the mathematical foundations of machine learning and their implementation in Rust, these prompts are designed to explore the concepts introduced in this chapter with technical depth and precision. Each prompt challenges you to think critically and apply mathematical principles using Rust, ensuring a comprehensive grasp of the material.
</p>

- <p style="text-align: justify;">Explain the role of linear algebra in machine learning, particularly in the context of operations on high-dimensional data. How do matrix operations like multiplication, inversion, and decomposition underpin key algorithms such as PCA and linear regression? Illustrate with examples implemented in Rust.</p>
- <p style="text-align: justify;">Discuss the concept of eigenvalues and eigenvectors in the context of machine learning. How are they used in techniques such as PCA and spectral clustering? Provide a detailed explanation of how to compute them using Rust, and explore their significance in dimensionality reduction.</p>
- <p style="text-align: justify;">Provide an in-depth analysis of gradient descent and its variants (e.g., stochastic gradient descent, mini-batch gradient descent). How do these methods optimize loss functions in machine learning models, and what are the trade-offs between them? Implement these algorithms in Rust and discuss their performance.</p>
- <p style="text-align: justify;">Explore the role of convexity in optimization problems. Why is convexity important in machine learning, particularly in the context of gradient descent? Provide examples of convex and non-convex optimization problems, and discuss how Rust can be used to solve them.</p>
- <p style="text-align: justify;">Discuss the importance of probability distributions in machine learning, particularly in modeling data and making predictions. How do distributions like the Gaussian and Bernoulli play a role in algorithms like Naive Bayes and logistic regression? Implement probabilistic models in Rust and apply them to real-world data.</p>
- <p style="text-align: justify;">Explain Bayes' theorem and its application in machine learning, particularly in the context of Bayesian inference and probabilistic graphical models. Provide a detailed implementation of Bayesian inference in Rust, and discuss its applications in machine learning.</p>
- <p style="text-align: justify;">Analyze the geometric interpretation of linear regression. How does the least squares method relate to projecting data onto a lower-dimensional space? Implement a linear regression model in Rust and discuss the significance of the normal equation in finding the optimal solution.</p>
- <p style="text-align: justify;">Explore the concept of regularization in linear models, particularly in the context of Ridge and Lasso regression. How does regularization help prevent overfitting, and what are the trade-offs involved? Implement regularized linear models in Rust and compare their performance on different datasets.</p>
- <p style="text-align: justify;">Discuss the importance of numerical methods in machine learning, particularly when analytical solutions are not feasible. How do algorithms like Newton-Raphson and the bisection method help in solving optimization problems? Implement these methods in Rust and apply them to optimize a complex loss function.</p>
- <p style="text-align: justify;">Provide an in-depth explanation of combinatorial optimization and its applications in machine learning. How are techniques like dynamic programming and greedy algorithms used to solve problems such as the traveling salesman problem or knapsack problem? Implement combinatorial optimization algorithms in Rust and discuss their performance.</p>
- <p style="text-align: justify;">Analyze the role of graph theory in machine learning, particularly in the context of data structures and relationships. How are graphs used in algorithms like decision trees, Bayesian networks, and clustering? Implement graph-based algorithms in Rust and explore their applications in machine learning.</p>
- <p style="text-align: justify;">Discuss the significance of the dot product and its applications in machine learning, particularly in the context of measuring similarity and projection. How is the dot product used in algorithms like SVMs and neural networks? Implement the dot product operation in Rust and apply it to a machine learning task.</p>
- <p style="text-align: justify;">Explore the concept of matrix factorization and its applications in machine learning, particularly in recommendation systems and latent factor models. How does Singular Value Decomposition (SVD) work, and how is it implemented in Rust? Discuss its role in dimensionality reduction and collaborative filtering.</p>
- <p style="text-align: justify;">Provide a detailed explanation of partial derivatives and their role in backpropagation in neural networks. How do partial derivatives help in computing gradients and updating weights? Implement backpropagation in Rust and analyze the challenges involved in optimizing a deep neural network.</p>
- <p style="text-align: justify;">Discuss the role of probability density functions (PDFs) in machine learning, particularly in the context of estimating the distribution of data. How are PDFs used in techniques like kernel density estimation and Gaussian mixture models? Implement a PDF in Rust and apply it to a machine learning problem.</p>
- <p style="text-align: justify;">Analyze the importance of the covariance matrix in understanding the relationships between features in a dataset. How is the covariance matrix used in PCA and linear discriminant analysis (LDA)? Implement the computation of a covariance matrix in Rust and apply it to a dataset for feature extraction.</p>
- <p style="text-align: justify;">Explore the concept of numerical differentiation and its applications in machine learning, particularly in the context of optimizing non-linear functions. How is numerical differentiation used in gradient-based optimization methods? Implement numerical differentiation in Rust and discuss its accuracy compared to analytical methods.</p>
- <p style="text-align: justify;">Discuss the application of graph theory in clustering algorithms, particularly in techniques like spectral clustering and community detection. How do graphs help in representing and analyzing the structure of data? Implement a graph-based clustering algorithm in Rust and apply it to a complex dataset.</p>
- <p style="text-align: justify;">Provide an in-depth analysis of combinatorics and its role in machine learning, particularly in the context of feature selection and model complexity. How do combinatorial techniques help in solving problems like feature subset selection? Implement a combinatorial optimization technique in Rust and apply it to a machine learning task.</p>
- <p style="text-align: justify;">Explore the concept of matrix inversion and its significance in solving systems of linear equations. How is matrix inversion used in algorithms like linear regression and Kalman filters? Implement matrix inversion in Rust and discuss its computational challenges and alternatives like matrix decomposition.</p>
<p style="text-align: justify;">
Each prompt encourages you to explore, analyze, and apply mathematical principles, pushing you toward mastery in both theory and practice. Embrace these challenges as opportunities to grow and innovate, knowing that the knowledge you gain will be invaluable in your journey as a machine learning practitioner.
</p>

## 3.7.2. Hands On Practices
<p style="text-align: justify;">
These exercises are designed to be challenging and in-depth, requiring you to apply both mathematical theory and Rust programming skills to solve complex machine learning problems.
</p>

#### **Exercise 3.1:** Advanced Matrix Operations and Decomposition Techniques in Rust
- <p style="text-align: justify;"><strong>Task</strong>: Implement advanced matrix operations in Rust, including matrix multiplication, transposition, and inversion. Focus on implementing matrix decomposition techniques such as LU decomposition and Singular Value Decomposition (SVD). Apply these techniques to perform Principal Component Analysis (PCA) on a high-dimensional dataset.</p>
- <p style="text-align: justify;"><strong>Challenges</strong>: Implement the matrix operations and decomposition from scratch without relying on external libraries, ensuring numerical stability and efficiency. Apply PCA to reduce the dimensionality of a complex dataset and analyze the results in terms of variance explained and data compression.</p>
#### **Exercise 3.2:** Implementing and Optimizing Gradient Descent Algorithms in Rust
- <p style="text-align: justify;"><strong>Task</strong>: Develop multiple variants of the gradient descent algorithm in Rust, including batch gradient descent, stochastic gradient descent (SGD), and mini-batch gradient descent. Apply these algorithms to train a deep neural network on a non-linear regression problem.</p>
- <p style="text-align: justify;"><strong>Challenges</strong>: Implement the gradient descent algorithms with a focus on optimizing the learning rate, convergence criteria, and computational efficiency. Analyze the performance of each variant in terms of speed, accuracy, and convergence behavior. Discuss the trade-offs between the different methods and how they impact the training of complex models.</p>
#### **Exercise 3.3:** Building Probabilistic Models Using Rust
- <p style="text-align: justify;"><strong>Task</strong>: Implement a probabilistic model, such as a Naive Bayes classifier or a Gaussian Mixture Model (GMM), in Rust. Apply the model to a real-world classification problem, such as sentiment analysis or spam detection.</p>
- <p style="text-align: justify;"><strong>Challenges</strong>: Implement the probabilistic model from scratch, including the calculation of probability distributions, likelihoods, and posterior probabilities. Evaluate the model's performance using cross-validation and analyze its strengths and weaknesses compared to other classification techniques. Optimize the implementation for large datasets, focusing on computational efficiency and memory management.</p>
#### **Exercise 3.4:** Numerical Optimization and Root-Finding Algorithms in Rust
- <p style="text-align: justify;"><strong>Task</strong>: Implement numerical optimization techniques in Rust, including the Newton-Raphson method and the bisection method, to solve complex non-linear equations. Apply these methods to optimize a machine learning model's loss function, such as in logistic regression or support vector machines.</p>
- <p style="text-align: justify;"><strong>Challenges</strong>: Implement the optimization algorithms with a focus on ensuring convergence and handling edge cases where the function may not be well-behaved. Compare the performance of the different methods in terms of speed, accuracy, and robustness, and discuss the implications of numerical optimization in the context of machine learning.</p>
#### **Exercise 3.5:** Graph Theory and Combinatorial Optimization in Rust
- <p style="text-align: justify;"><strong>Task</strong>: Implement graph-based algorithms in Rust, such as Dijkstra’s shortest path algorithm and Kruskal’s minimum spanning tree algorithm. Apply these algorithms to a machine learning problem, such as clustering or feature selection, where the structure of the data can be represented as a graph.</p>
- <p style="text-align: justify;"><strong>Challenges</strong>: Implement the graph algorithms from scratch, focusing on efficiency and scalability for large graphs. Apply the algorithms to a complex dataset, such as a social network or a biological network, and analyze the results in terms of clustering quality, feature selection, or community detection. Discuss the role of combinatorial optimization in solving these problems and the challenges of applying graph theory in machine learning.</p>
<p style="text-align: justify;">
By completing these tasks, you will develop a deep understanding of the mathematical foundations of machine learning and gain hands-on experience in implementing these concepts in Rust. Embrace the difficulty of these exercises as an opportunity to grow your expertise and prepare yourself for tackling real-world challenges in machine learning.
</p>
