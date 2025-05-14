# Hierarchical graph-based machine learning model for optimization of three-dimensional braced steel frame

Chi-tathon Kupwiwat $ {}^{ \star  } $ (C), Kazuki Hayashi (C), Makoto Ohsaki

Department of Architecture and Architectural Engineering, Kyoto University, Kyoto, 615-8540, Japan

## ARTICLEINFO

JEL classification:

E2110B

Keywords:

Topology optimization

Hierarchical graph-based machine learning

Reinforcement learning

Graph neural network

Braced steel frame

## A B S T R A C T

This paper proposes a method for topology optimization of three-dimensional braced steel frames under static seismic loads, aiming to minimize structural volume while satisfying response constraints. The method consists of a proposed hierarchical graph-based machine learning model that sequentially embeds graph representations of structural elements to create a comprehensive lumped mass model suitable for seismic analysis and optimization of building frames. The proposed model is utilized as a reinforcement learning agent, a class of machine learning, to observe the current building frame configurations, modify the frame to improve its performance, and adjust the model parameters using the obtained reward function. Numerical results demonstrate that the proposed graph-based model can improve its performance better than benchmark of a graph neural network utilized in previous research. When applied to three cases of unseen large three-dimensional building frames, the trained agent with the proposed model outperforms the genetic algorithm and simulated annealing in the optimization task with 11-46% lower objective function while utilizing the same computational cost. The generality, performance, and computational efficiency of the agent indicate its applicability to applied to the optimization of three-dimensional frames.

## 1. Introduction

Optimization of discrete structures such as frames and trusses aims to minimize/maximize the objective function, which is determined by the structural volume or performance under various constraints (Christensen and Klarbring, 2010), by finding the best structural configurations. When these configurations are nodal connectivity in addition to cross-sectional properties of structural elements, the optimization problem is classified as topology optimization problem. Examples of topology optimization of discrete structures are shown in Ohsaki and Hayashi (2017), Raphami et al. (Rahami et al., 2008), and Cheng and Guo (1997).

To obtain optimal solutions of building structures, the optimization procedures can be classified into the gradient-based approaches of iteratively updating the design using gradient information (Boggs and Tolle, 1995; Putresza and Kolakowski, 1998) and the methods based on metaheuristics, which are further classified into methods of improving a single solution stochastically and/or based on local search (Robbins and Monro, 1951) and the population-based methods iteratively improving multiple solutions using evolutionary operators (Poli et al., 2007; Kaveh and Zolghadr, 2014; Deb et al., 2002). When the design variables are selected from the sets of discrete values, topology optimization problem of three-dimensional braced steel frame can pose a real-world design challenge, because the gradients of the objective and constraint functions with respect to the design variables are not available (Ohsaki, 2016); therefore, metaheuristics are preferred. Stochastic-based, and population-based metaheuristic approaches such as simulated annealing (SA) (Kirkpatrick et al., 1983) and genetic algorithm (GA) (Holland, 1992), respectively, have been successfully applied. SA is a local search algorithm for stochastically improving solutions, whereas GA is an algorithm that generates multiple solutions, i.e., population, and selects the best solutions to generate population for the next optimization step. Since both SA and GA are heuristic algorithms and the structural optimization problems are generally nonconvex, the obtained solutions are not guaranteed to be globally optimal. The algorithms also require multiple parameters tuning, such as crossover and mutation rate in GA, to improve the results. These heuristic nature in parameter setting require the user to run the algorithms multiple times to obtain an optimal solution with good accuracy. The already found optimal settings and parameters in these algorithms for one problem may not perform well in other problems (Osaba et al., 2021; Rezk et al., 2024). Therefore,

---

* Corresponding author.

E-mail address: se.kupwiwat@archi.kyoto-u.ac.jp (C.-t. Kupwiwat).

---

## Abbreviations

GA : Genetic algorithm

GCN : Graph convolutional network

HEBG : Hierarchical embedding method of building graph

SA : Simulate Annealing

RL : Reinforcement learning when encountering many problems, such as in the initial design phase of the buildings, utilizing these algorithms may not be feasible.

In recent years, many research results have indicated that machine learning (ML) has become more effective in optimizing building structures in terms of both accuracy and computational cost (Tamura et al., 2018; Naranjo-Pérez et al., 2020; Sakaguchi et al., 2021; Hayashi and Ohsaki, 2021a). ML is a method to train a machine or agent to do specific tasks (Goodfellow et al., 2016). It can be roughly classified into unsupervised learning, which does not need labeling of training data; supervised learning, which requires training data labeling; and reinforcement learning (RL), which collects and labels data during the training process. Among these methods, RL can be seamlessly integrated into the optimization algorithms and utilized to find better solutions in each optimization step. Kupwiwat and Yamamoto (2020), and Yama-moto and Kupwiwat (2023) explored the possibility of application of RL in simple geometry optimization of lattice shells. However, their methods of encoding training data are tabular-based that are ineffective when the building structure becomes larger or differs from those employed during the training process of RL.

Graph is a data structure consisting of nodes (vertices) and edges that connect pairs of nodes. Graphs can be represented in a matrix form that allows the processing of this type of data efficiently, as shown by Kaveh et al. (Kaveh, 1992) which explored graph theory and its application for structural analysis and optimization. Hayashi and Ohsaki (2021b) proposed a novel method to represent truss structures as graph data and combined the representation with RL, enabling the model to optimize the truss effectively even though its connectivity differs from the training data. The graph representation is also applicable to other ML methods as shown in the works by Cao et al. (2024), Dang et al. (Dang and Nguyen, 2024), Ororbia et al. (Ororbia and Warn, 2024), and Park et al. (2024) which utilized graph representation for the truss classification, structural damage detection, discrete structural generation, and the constrained optimization of plane truss, respectively. Kupwiwat et al. (2023) showed that the optimal topology of planar braced frames can be effectively obtained using a method that combines RL utilizing graph convolutional networks (GCNs) (Kipf and Welling, 2017) and graph representation. However, extending their method is challenging for a real-world three-dimensional building frame due to its large number of structural elements, which hinders the GCN agent from learning an effective policy within the vast action space (Duan et al., 2022). One possible solution to this problem is to embed the original graph data into a smaller one using a dimensionality reduction method such as principal component analysis (Pearson, 1901) and vector quantization (Gray, 1984). While embedding models such as graph fully convolutional network (graph U-net) (Gao and Ji, 2019) attempt to address this by pooling and reducing graph size, they do not hierarchically reconstruct the graph based on domain-specific knowledge about the graph, which is vital for optimizing complex three-dimensional frames.

To overcome these limitations, this paper proposes a novel hierarchical embedding-based building graph (HEBG) model that integrates structural embedding into a lumped mass model employed in structural analysis. This approach reduces computational complexity and systematically guides the model to retain essential structural details, enabling versatile and inexpensive optimization process. The proposed model is integrated with RL and graph representation for solving the topology optimization problem of the three-dimensional frames subjected to static seismic loads. The objective function to be minimized is the structural volume under constraints assigned for structural responses, and an approximate optimal solution is found using a greedy algorithm adding a brace or increasing the size of a structural element at each step of optimization. The numerical results indicate that the proposed model has superior performance over a normal GCN model in the training phase. Moreover, the trained RL agent using HEBG outperforms GA and SA when applied to optimization of three-dimensional building frames.

The rest of the paper is organized as follows: Section 2 details the optimization problem and the approach for calculating seismic loads. Sections 3 and 4 provide a detailed overview of the proposed HEBG and machine learning techniques employed in this study. Section 5 outlines the training process of the HEBG model. Section 6 presents numerical examples of the training and testing phases, demonstrating the learning efficiency and quality of optimization results of the proposed HEBG in comparison to the existing methods.

## 2. Optimization problem

### 2.1. Outlines

The optimization problem is formulated as a combinatorial problem where the design variables are topology of braces and properties of structural elements consisting of columns, beams, and braces in the building frame. The frame is subjected to long-term loads and short-term loads, respectively. The long-term loads consist of the dead load (self-weight) and the live load computed from the building function as an office. The short-term loads are the design seismic loads in addition to the long-term loads. Optimization of a three-dimensional building frame starts from the smallest column and beam sections without braces. Note that this initialized structure cannot withstand the long-term and short-term loads, i.e., it violates the constraints on structural responses defined by stresses and displacements. During the optimization process, an RL agent determines the policy for the action of either adding a brace or increasing the section of an existing structural element. The optimization process ends when all constraints on structural responses are satisfied.

### 2.2. Objective function

The optimal brace placements and section sizes of structural elements of steel frames are obtained by minimizing the objective function formulated as the total structural volume of columns, beams, and braces.

Let $ {c}_{\mathrm{f}} $ and $ {b}_{\mathrm{f}} $ be the numbers of columns and beams in the three-dimensional building frame, respectively. Cross-sectional properties of columns and beams are selected from the predetermined lists of $ {n}_{\mathrm{c}} $ and $ {n}_{\mathrm{b}} $ candidate sections, respectively. Vectors representing the properties of columns and beams are as follows:

$$
\mathbf{c} = \left( {{c}_{1},\ldots ,{c}_{{c}_{\mathrm{f}}}}\right)  \in  {\mathbb{R}}^{{c}_{\mathrm{f}}} \tag{1}
$$

$$
\mathbf{b} = \left( {{b}_{1},\ldots ,{b}_{{b}_{\mathrm{f}}}}\right)  \in  {\mathbb{R}}^{{b}_{\mathrm{f}}} \tag{2}
$$

where the indices of the predetermined cross-sectional properties of columns and beams are $ {c}_{i \in  \left\{  {1,\ldots ,{c}_{\mathrm{f}}}\right\}  } \in  \left\{  {0,1,\ldots ,{n}_{\mathrm{c}}}\right\} $ and $ {b}_{i \in  \left\{  {1,\ldots ,{b}_{\mathrm{f}}}\right\}  } \in  \{ 0,1 $ , $ \left. {\ldots ,{n}_{\mathrm{b}}}\right\} $ , respectively.

Let $ {r}_{\mathrm{f}},{r}_{\mathrm{t}} $ , and $ {n}_{\mathrm{r}} $ be the number of braces that can exist in the building frame, the number of brace types, and the number of predetermined cross-sectional properties of braces, respectively. A matrix representing the braces with their components $ {r}_{i \in  \left\{  {1,\ldots ,{r}_{\mathrm{t}}}\right\}  , j \in  \left\{  {1,\ldots ,{r}_{\mathrm{t}}}\right\}  } \in  \left\{  {0,1,\ldots ,{n}_{\mathrm{r}}}\right\} $ is as follows:

$$
\mathbf{r} = \left\lbrack  \begin{matrix} {r}_{1,1} & \cdots & {r}_{1,{r}_{\mathrm{t}}} \\  \vdots &  \ddots  & \vdots \\  {r}_{{r}_{\mathrm{f}},1} & \cdots & {r}_{{r}_{\mathrm{f}},{r}_{\mathrm{t}}} \end{matrix}\right\rbrack   \in  {\mathbb{R}}^{{r}_{\mathrm{f}} \times  {r}_{\mathrm{t}}} \tag{3}
$$

where $ {r}_{i, j} $ is either 0 or 1 which indicates existence/non-existence of the $ j $ th brace type at the $ i $ th location in the frame. Note that each row in $ \mathbf{r} $ has at most one non-zero entry and no brace exists if all entries are 0 .

Columns and beams are modeled by three-dimensional beam elements with 12 degrees of freedom (DoFs), which are translations in $ x, y $ , and $ z $ -axes and rotations around $ x, y $ , and $ z $ -axes at both ends of the element, and braces are three-dimensional truss elements with $ 6\mathrm{{DoFs}} $ ), including of translations in $ x, y $ , and $ z $ -axes at both ends of the element. The internal stress can be obtained from the local displacements of each element after solving the stiffness equation $ \mathbf{{Ku}} = \mathbf{p} $ where $ \mathbf{p},\mathbf{K} $ , and $ \mathbf{u} $ are the nodal load vector, the global stiffness matrix, and the global displacement vector, respectively. The ratio of internal stress to the yield stress, denoted by $ {\sigma }_{i} $ , of element $ i $ is computed similarly to Lagaros et al. (2005) as follows:

$$
{\sigma }_{i} = \left\{  \begin{array}{lll} \frac{\left| {\sigma }_{{v}_{1}}\right| }{{\bar{\sigma }}_{{v}_{1}}} + \frac{\left| {\sigma }_{{v}_{2}}\right| }{{\bar{\sigma }}_{{v}_{2}}} + \frac{\left| {\sigma }_{{v}_{3}}\right| }{{\bar{\sigma }}_{{v}_{3}}} &  : & \frac{\left| {\sigma }_{{v}_{1}}\right| }{{\bar{\sigma }}_{{v}_{1}}} \leq  {0.15} \\  \frac{\left| {\sigma }_{{v}_{1}}\right| }{{0.6}{\bar{\sigma }}_{{v}_{1}}} + \frac{\left| {\sigma }_{{v}_{2}}\right| }{{\bar{\sigma }}_{{v}_{2}}} + \frac{\left| {\sigma }_{{v}_{3}}\right| }{{\bar{\sigma }}_{{v}_{3}}} &  : & \frac{\left| {\sigma }_{{v}_{1}}\right| }{{\bar{\sigma }}_{{v}_{1}}} > {0.15} \end{array}\right.  \tag{4}
$$

where $ {\sigma }_{{v}_{1}} $ is the axial stress, and $ {\sigma }_{{v}_{2}} $ and $ {\sigma }_{{v}_{3}} $ are the maximum bending stresses in the strong and weak axes of the section, respectively, $ {\bar{\sigma }}_{{v}_{1}},{\bar{\sigma }}_{{v}_{2}} $ , and $ {\bar{\sigma }}_{{v}_{3}} $ denote their upper bounds.

The functions of material volume of columns, beams, and braces are denoted by $ {V}_{\mathrm{c}}\left( \mathbf{c}\right) ,{V}_{\mathrm{b}}\left( \mathbf{b}\right) $ , and $ {V}_{\mathrm{r}}\left( \mathbf{r}\right) $ , respectively. The optimization problem to minimize the structural volume denoted by $ V\left( {\mathbf{c},\mathbf{b},\mathbf{r}}\right) $ subject to stress, inter-story drift angle, and beam deflection constraints is formulated as:

$$
\text{minimize}\;V\left( {\mathbf{c},\mathbf{b},\mathbf{r}}\right)  = {V}_{\mathrm{c}}\left( \mathbf{c}\right)  + {V}_{\mathrm{b}}\left( \mathbf{b}\right)  + {V}_{\mathrm{r}}\left( \mathbf{r}\right)
$$

subject to $ {\sigma }_{\max }^{\mathrm{L}}\left( {\mathbf{c},\mathbf{b},\mathbf{r}}\right)  \leq  1 $

$$
{\sigma }_{\max }^{\mathrm{S}}\left( {\mathbf{c},\mathbf{b},\mathbf{r}}\right)  \leq  1 \tag{5}
$$

$$
{\theta }_{\max }\left( {\mathbf{c},\mathbf{b},\mathbf{r}}\right)  \leq  1
$$

$$
{\delta }_{\max }\left( {\mathbf{c},\mathbf{b},\mathbf{r}}\right)  \leq  1
$$

where $ {\sigma }_{\max }^{\mathrm{L}} $ and $ {\sigma }_{\max }^{\mathrm{S}} $ are the maximum stress ratios at the element ends among all members under the long-term loads (dead and live loads) and short-term loads (seismic loads in addition to long-term loads), respectively. $ {\theta }_{\max } $ is the maximum ratio of inter-story drift angles to the upper bound 1/200 among all stories due to the short-term loads, as specified by the Eurocode 3 (European Committee for Standardization, 2007). $ {\delta }_{\max } $ is the maximum ratio of beam deflections among all beams defined as follows:

$$
{\delta }_{\max } = \mathop{\max }\limits_{i}\left( {\frac{{300}\left| {d}_{i}^{\mathrm{b}}\right| }{{L}_{i}^{\mathrm{b}}},1}\right)  \tag{6}
$$

where $ {d}_{i}^{\mathrm{b}} $ and $ {L}_{i}^{\mathrm{b}} $ are the deflection at the center and the length of beam $ i $ , respectively, i.e., the upper bound of deflection is 1/300 of the beam length (European Committee for Standardization, 2007).

### 2.3. External loads

In the long-term loading scenario, the frame is subjected to a combined dead load of $ {5000}\mathrm{\;N} $ per square meter $ \left( {\mathrm{N}/{\mathrm{m}}^{2}}\right) $ and live load of $ {1800}\mathrm{\;N}/{\mathrm{m}}^{2} $ in the negative $ z $ -direction (i.e., vertical direction) which is distributed to nodes in the frame model as illustrated in Figure (Fig.) 1 (A). Note that each beam is divided into two elements to consider its mid-point deflection. Short-term loads are obtained using the response spectrum approach (Chopra, 2007). We consider four cases where the frame is subjected to seismic design loads, computed using the floor weight of $ {5800}\mathrm{\;N}/{\mathrm{m}}^{2} $ , in four directions on the $ x - y $ plane $ ( + x, - x, + y $ , $ - y $ ), respectively, as shown in Fig. 1B. Note that the long-term loads are applied before applying the short-term seismic design loads.

## 3. Graph embedding

### 3.1. Graph representation and graph convolutional network

A graph is a type of data structure that has nodes connected by edges. Note that nodes contain data indicated as features and edges represent the connectivity or relationship between nodes. A graph having $ n $ nodes and $ h $ features in each node can be represented using a node feature matrix $ \mathbf{N} \in  {\mathbb{R}}^{n \times  h} $ . An adjacency matrix $ {\mathbf{A}}_{\mathrm{n}} \in  {\mathbb{R}}^{n \times  n} $ and a degree matrix $ \mathbf{D} \in  {\mathbb{R}}^{n \times  n} $ represent the nodal connections and the number of connected elements at each node, respectively. Previous research by Kupwiwat et al. (2023) showed that a two-dimensional building frame can be represented as a graph where the graph nodes and edges represent the structural elements and connectivity between these elements, respectively, which are subsequently transformed into the state data in their RL framework. GCN is a class of neural network that can process graph representations. Let $ \mathbf{I} \in  {\mathbb{R}}^{n \times  n} $ and $ {\mathbf{D}}^{-1/2} $ be the identity matrix and the inverse of the matrix $ {\mathbf{D}}^{1/2} $ satisfying $ {\mathbf{D}}^{1/2}{\mathbf{D}}^{1/2} = \mathbf{D} $ , respectively. A GCN computing unit is referred to as a layer that takes $ \mathbf{N} $ and a normalized $ \mathbf{A} $ denoted as $ \widetilde{\mathbf{A}} = {\mathbf{D}}^{-1/2}\left\lbrack  {\mathbf{A} + \mathbf{I}}\right\rbrack  {\mathbf{D}}^{-1/2} $ as inputs to compute the embedded node signal $ {\mathbf{N}}^{\prime } $ as follows:

$$
{\mathbf{N}}^{\prime } = \mu \left( {\mathbf{N},\mathbf{A}}\right)  = \mu \left( {\widetilde{\mathbf{A}}\mathbf{{Nw}}}\right)  \tag{7}
$$

where $ \mu \left( \cdot \right) $ denotes a nonlinear activation function, and $ \mathbf{w} \in  {\mathbb{R}}^{h \times  g} $ is an internal weight matrix in the GCN layer with $ g $ filters or feature maps (Kipf and Welling, 2017). Note that multiple GCN layers can be stacked and the next layer can take the embedded node signal $ {\mathbf{N}}^{\prime } $ from the previous layer as one of the inputs instead of the original $ \mathbf{N} $ .

### 3.2. Proposed hierarchical graph embedding

To simplify a three-dimensional building frame into a smaller graph of a lumped mass model, this research proposes the HEBG model that takes the graph representing the data of \{braces, beams, columns, floors\} as input and sequentially embeds the graph into the data of \{beams, columns, floors\}, \{columns, floors\}, and \{floors\}, utilizing specific graph pooling operation for each data. After the embedded graph of building floors (i.e., lump-mass) is obtained, the data of \{columns, floors\}, \{beams, columns, floors\}, and \{braces, beams, columns, floors\} are sequentially restored. The RL agent uses the restored data of \{braces, beams, columns, floors\} as output which determines the modification of the building frame to enhance its structural performance against the external loads. For a building frame structure, this process of pooling can be determined based on types of structural elements (floor, column, beam, brace) as shown in Fig. 2. We utilize the proposed model to make the RL agent, which is trained to optimize the three-dimensional braced steel frames to minimize structural volume under constraints on structural responses against seismic loads.

Let $ \mathcal{G} $ be a graph of building frame elements with $ {f}_{\mathrm{f}} $ floors except the ground floor. The node feature matrices of floors, columns, beams, and braces with $ h $ features, $ {\mathbf{N}}^{\mathrm{f}} \in  {\mathbb{R}}^{{f}_{\mathrm{f}} \times  h},{\mathbf{N}}^{\mathrm{c}} \in  {\mathbb{R}}^{{c}_{\mathrm{f}} \times  h},{\mathbf{N}}^{\mathrm{b}} \in  {\mathbb{R}}^{{b}_{\mathrm{f}} \times  h} $ , and $ {\mathbf{N}}^{\mathrm{r}} \in  {\mathbb{R}}^{{r}_{\mathrm{f}} \times  h} $ , respectively, are defined as follows:

$$
{\mathbf{N}}^{\mathrm{f}} = \left\lbrack  \begin{matrix} {n}_{1,1}^{\mathrm{f}} & \cdots & {n}_{1, h}^{\mathrm{f}} \\  \vdots & \vdots & \vdots \\  {n}_{{f}_{\mathrm{f}},1}^{\mathrm{f}} & \cdots & {n}_{{f}_{\mathrm{f}}, h}^{\mathrm{f}} \end{matrix}\right\rbrack  ,{\mathbf{N}}^{\mathrm{c}} = \left\lbrack  \begin{matrix} {n}_{c,1}^{\mathrm{c}} & \cdots & {n}_{1, h}^{\mathrm{c}} \\  \vdots & \vdots & \vdots \\  {n}_{{c}_{\mathrm{r}},1}^{\mathrm{c}} & \cdots & {n}_{{c}_{\mathrm{r}}, h}^{\mathrm{c}} \end{matrix}\right\rbrack  ,{\mathbf{N}}^{\mathrm{b}} = \left\lbrack  \begin{matrix} {n}_{1,1}^{\mathrm{b}} & \cdots & {n}_{1, h}^{\mathrm{b}} \\  \vdots & \vdots & \vdots \\  {n}_{{b}_{\mathrm{r}},1}^{\mathrm{b}} & \cdots & {n}_{{b}_{\mathrm{r}}, h}^{\mathrm{b}} \end{matrix}\right\rbrack  ,{\mathbf{N}}^{\mathrm{r}} = \left\lbrack  \begin{matrix} {n}_{1,1}^{\mathrm{r}} & \cdots & {n}_{1, h}^{\mathrm{r}} \\  \vdots & \vdots & \vdots \\  {n}_{{r}_{\mathrm{f}},1}^{\mathrm{r}} & \cdots & {n}_{{r}_{\mathrm{f}}, h}^{\mathrm{r}} \end{matrix}\right\rbrack
$$

![0196c354-f366-7182-8e00-4fe98707e99c_3_98_151_1554_713_0.jpg](images/0196c354-f366-7182-8e00-4fe98707e99c_3_98_151_1554_713_0.jpg)

Fig. 1. Long-term and short-term loads.

![0196c354-f366-7182-8e00-4fe98707e99c_3_192_935_1357_630_0.jpg](images/0196c354-f366-7182-8e00-4fe98707e99c_3_192_935_1357_630_0.jpg)

Fig. 2. Hierarchical pooling graph for building frames.

The node feature matrix $ \mathbf{N} \in  {\mathbb{R}}^{\left( {{f}_{\mathrm{f}} + {c}_{\mathrm{f}} + {b}_{\mathrm{f}} + {r}_{\mathrm{f}}}\right)  \times  h} $ that represents all nodes in the building frame is the block notation of $ {\mathbf{N}}^{\mathrm{f}},{\mathbf{N}}^{\mathrm{c}},{\mathbf{N}}^{\mathrm{b}} $ , and $ {\mathbf{N}}^{\mathrm{r}} $ , defined as

$$
\mathbf{N} = \left\lbrack  \begin{matrix} {\mathbf{N}}^{\mathrm{f}} \\  {\mathbf{N}}^{\mathrm{c}} \\  {\mathbf{N}}^{\mathrm{b}} \\  {\mathbf{N}}^{\mathrm{r}} \end{matrix}\right\rbrack   \tag{8}
$$

Let $ {\mathbf{A}}^{\mathrm{p},\mathrm{q}} $ denote the adjacency matrix that represents connectivity between element types $ \mathrm{p} $ and $ \mathrm{q} $ , using a value of $ \{ 0,1\} $ , where $ \mathrm{p},\mathrm{q} \in  \{ \mathrm{f},\mathrm{c} $ , $ \mathrm{b},\mathrm{r}\} $ . Let $ {\left( \bullet \right) }^{\mathrm{T}} $ denote a transpose operator. The adjacency matrix $ \mathbf{A} \in $ $ {\mathbb{R}}^{\left( {{f}_{\mathrm{f}} + {c}_{\mathrm{f}} + {b}_{\mathrm{f}} + {r}_{\mathrm{f}}}\right)  \times  \left( {{f}_{\mathrm{f}} + {c}_{\mathrm{f}} + {b}_{\mathrm{f}} + {r}_{\mathrm{f}}}\right) } $ that represents the connectivity of all elements in the building frame is expressed by block notation as

$$
\mathbf{A} = \left\lbrack  \begin{matrix} {\mathbf{A}}^{\mathrm{f},\mathrm{f}} & {\mathbf{A}}^{\mathrm{f},\mathrm{c}} & {\mathbf{A}}^{\mathrm{f},\mathrm{b}} & {\mathbf{A}}^{\mathrm{f},\mathrm{r}} \\  {\mathbf{A}}^{\mathrm{f},{\mathrm{c}}^{\mathrm{T}}} & {\mathbf{A}}^{\mathrm{c},\mathrm{c}} & {\mathbf{A}}^{\mathrm{c},\mathrm{b}} & {\mathbf{A}}^{\mathrm{c},\mathrm{r}} \\  {\mathbf{A}}^{\mathrm{f},{\mathrm{b}}^{\mathrm{T}}} & {\mathbf{A}}^{\mathrm{c},{\mathrm{b}}^{\mathrm{T}}} & {\mathbf{A}}^{\mathrm{b},\mathrm{b}} & {\mathbf{A}}^{\mathrm{b},\mathrm{r}} \\  {\mathbf{A}}^{\mathrm{f},{\mathrm{r}}^{\mathrm{T}}} & {\mathbf{A}}^{\mathrm{c},{\mathrm{r}}^{\mathrm{T}}} & {\mathbf{A}}^{\mathrm{b},{\mathrm{r}}^{\mathrm{T}}} & {\mathbf{A}}^{\mathrm{r},\mathrm{r}} \end{matrix}\right\rbrack   \tag{9}
$$

Utilizing this block notation, the hierarchical graph is embedded by removing entries in the node and adjacency matrices. Note that we sequentially remove entries of brace, beam, and column, respectively. To embed the signal of removed entries, we utilize a GCN layer with a sigmoid activation function and then remove the entries. This operation is denoted as pooling. To reconstruct the signal, the block notation of the node is made, using the current signal and the zero matrices with the same size as the signal to be reconstructed. This block notation is combined with the signal from the skip-connection operation and then passed to a GCN with the Rectified Linear Unit (ReLU) activation function. This operation is named un-pooling. Since the signal of the element could be lost when restored, we employed another GCN layer with a sigmoid activation function that passes the signal directly to un-pooling layers, denoted as skip-connection. Table 1 details the proposed hierarchical graph embedding operations, including pooling, un-pooling, and skip-connection, where

$$
{\mathbf{A}}^{\prime } = \left\lbrack  \begin{matrix} {\mathbf{A}}^{\mathrm{f},\mathrm{f}} & {\mathbf{A}}^{\mathrm{f},\mathrm{c}} & {\mathbf{A}}^{\mathrm{f},\mathrm{b}} \\  {\mathbf{A}}^{\mathrm{f},{\mathrm{c}}^{\mathrm{T}}} & {\mathbf{A}}^{\mathrm{c},\mathrm{c}} & {\mathbf{A}}^{\mathrm{c},\mathrm{b}} \\  {\mathbf{A}}^{\mathrm{f},{\mathrm{b}}^{\mathrm{T}}} & {\mathbf{A}}^{\mathrm{c},{\mathrm{b}}^{\mathrm{T}}} & {\mathbf{A}}^{\mathrm{b},\mathrm{b}} \end{matrix}\right\rbrack  ,{\mathbf{A}}^{\prime \prime } = \left\lbrack  \begin{matrix} {\mathbf{A}}^{\mathrm{f},\mathrm{f}} & {\mathbf{A}}^{\mathrm{f},\mathrm{c}} \\  {\mathbf{A}}^{\mathrm{f},{\mathrm{c}}^{\mathrm{T}}} & {\mathbf{A}}^{\mathrm{c},\mathrm{c}} \end{matrix}\right\rbrack   \tag{10}
$$

Let $ {\mu }_{1} $ and $ {\mu }_{2} $ denote Sigmoid and ReLU activation functions, respectively. $ {\mathbf{0}}^{\mathrm{c}},{\mathbf{0}}^{\mathrm{b}} $ , and $ {\mathbf{0}}^{\mathrm{r}} $ are zero matrices with size $ {\mathbb{R}}^{{c}_{\mathrm{f}} \times  {h}^{\prime }},{\mathbb{R}}^{{b}_{\mathrm{f}} \times  {h}^{\prime }} $ , and $ {\mathbb{R}}^{{r}_{\mathrm{f}} \times  {h}^{\prime }} $ , respectively, where $ {h}^{\prime } $ denotes the number of embedding features. $ \odot $ is the Hadamard product. Note that all adjacency matrices are normalized before passing into the GCN layers in HFBG, and $ {\mathbf{A}}^{\prime },{\mathbf{A}}^{\prime \prime },{\mathbf{A}}^{\mathrm{f},\mathrm{f}} $ can be derived from A. Each step in Table 1 can be considered as an approximation function for the HEBG model that takes $ \mathbf{N} $ and $ \mathbf{A} $ as input and returns $ {\mathbf{N}}^{\prime } $ as an output as

$$
{\mathbf{N}}^{\prime } = \operatorname{HEBG}\left( {\mathbf{N},\mathbf{A}}\right)  \tag{11}
$$

Fig. 3 illustrates the computation of HEBG which can be applied as a model for RL agents, in computational graph diagram format.

## 4. Reinforcement learning

This research combines the proposed HEBG model with RL for topology optimization of three-dimensional building frames. RL is a type of ML that trains agents to perform tasks in an environment using a reward signal. Note that the environment refers to the system where the RL agent is trained. In general, RL comprises a policy function to decide action, a reward signal to measure the outcome of the actions, and a value function to estimate the accumulated reward the agent will receive from doing actions, respectively (Sutton and Barto, 2018). The interaction between the RL agent and the environment is formulated using a Markov decision process (MDP) (Bellman, 1954, 1957).

Fig. 4 illustrates the MDP used in this research. In a discrete step $ t $ , the agent observes a state $ {S}_{t} $ as graph representing the structure (Fig. 4A), and takes an action $ {\mathrm{A}}_{t} $ to adjust the building frame (Fig. 4B). The property of the building frame is modified by actions of the agent, and the changes in structural responses and objective function are computed accordingly (Fig. 4C). The agent receives a reward $ {\mathrm{R}}_{t + 1} $ , and observes the next state $ {S}_{t + 1} $ . Note that the RL agent keeps interacting with the environment until all constraints on structural responses are satisfied. Similar to our previous research on two-dimensional topology optimization (Kupwiwat et al., 2023), this research utilizes the deep deterministic policy gradient (DDPG) (Lillicrap et al., 2016) that has a policy function (actor network) and a value function (critic network) to determine the probability of taking an action in a state and estimate the obtained reward from the probability, respectively.

## 5. Machine learning and hierarchical embedding of building graph

### 5.1. State

The state utilizes the representation where node feature matrix $ \mathbf{N} $ and adjacency matrix $ \mathbf{A} $ represent structure and connectivity, respectively. The row $ i $ in the node feature matrix contains the entries of structural node $ i $ as shown in Table 2. Table 3 shows the values in A. Note that the two end nodes of a member are designated as start node and end node, and the node at the center of beam is called center node.(x, y, z)and $ \left( {v}_{1}\right. $ , $ {v}_{2},{v}_{3} $ ) refer to the global and local coordinates, respectively.

### 5.2. Agent with HEBG

In order to incorporate graph representation in Section 5.1, the policy and value functions utilize HEBG for their computation. The policy function takes state data defined using $ \mathbf{N} $ and $ \mathbf{A} $ as input to compute the output signal for the action $ \widehat{\pi } $ defined as follows:

$$
\mathbf{\pi }\left( {\mathbf{N},\mathbf{A}}\right)  = \acute{\mathbf{\pi }} \in  {\mathbb{R}}^{\left( {{f}_{\mathrm{f}} + {c}_{\mathrm{f}} + {b}_{\mathrm{f}} + {r}_{\mathrm{f}}}\right)  \times  1} \tag{12a}
$$

$$
\widehat{\mathbf{\pi }} = {\mu }_{1}\left( {\operatorname{HEBG}\left( {\mathbf{N},\mathbf{A}}\right) ,\mathbf{A}}\right)  \tag{12b}
$$

Table 1

HEBG operations.

<table><tr><td>step</td><td>inputs</td><td>operations</td><td>outputs</td><td>step</td><td>inputs</td><td>operations</td><td>outputs</td></tr><tr><td>0.</td><td>$ \mathrm{N} $ A</td><td>pooling $ {\mathbf{N}}_{1}^{0} = {\mu }_{1}\left( {\mathbf{N},\mathbf{A}}\right) $ $ {\mathbf{N}}_{2}^{0} = {\mu }_{1}\left( {\mathbf{N},\mathbf{A}}\right) $ $ {\mathbf{N}}_{2}^{0} \odot  {\mathbf{N}}_{1}^{0} $ $ {\mathbf{N}}^{0} = {\mathbf{N}}_{1}^{0} + {\mathbf{N}}_{2}^{0} $ Remove entries of $ {\mathbf{N}}^{\mathrm{r}} $ from $ {\mathbf{N}}^{0} $</td><td>$ {\mathbf{N}}^{0} $ $ {\mathbf{N}}_{1}^{0} $</td><td>3.</td><td>$ {\mathrm{N}}^{2} $ $ {\mathbf{A}}^{\mathrm{f},\mathrm{f}} $ $ {\mathrm{N}}_{1}^{2} $</td><td>un-pooling $ {\mathbf{N}}^{3} = {\mu }_{2}\left( {{\mathbf{N}}^{2},{\mathbf{A}}^{\mathrm{f},\mathrm{f}}}\right) $ $ {\mathbf{N}}^{3} = \left\lbrack  \begin{matrix} {\mathbf{N}}^{3} \\  {\mathbf{0}}^{\mathrm{c}} \end{matrix}\right\rbrack $ $ {\mathbf{N}}^{3} = {\mathbf{N}}^{3} + {\mathbf{N}}_{1}^{2} $ ; skip-connection</td><td>$ {\mathrm{N}}^{3} $</td></tr><tr><td>1.</td><td>$ {\mathbf{N}}^{0} $ $ {\mathbf{A}}^{\prime } $</td><td>pooling $ {\mathbf{N}}_{1}^{1} = {\mu }_{1}\left( {{\mathbf{N}}^{0},{\mathbf{A}}^{\prime }}\right) $ $ {\mathbf{N}}_{2}^{1} = {\mu }_{1}\left( {{\mathbf{N}}^{0},{\mathbf{A}}^{\prime }}\right) $ $ {\mathbf{N}}_{2}^{1} \odot  {\mathbf{N}}_{1}^{1} $ $ {\mathbf{N}}^{1} = {\mathbf{N}}_{1}^{1} + {\mathbf{N}}_{2}^{1} $ Remove entries of $ {\mathbf{N}}^{\mathrm{b}} $ from $ {\mathbf{N}}^{1} $</td><td>$ {\mathbf{N}}^{1} $ $ {\mathbf{N}}_{1}^{1} $</td><td>4.</td><td>$ {\mathrm{N}}^{3} $ $ {\mathbf{A}}^{\prime \prime } $ $ {\mathbf{N}}_{1}^{1} $</td><td>un-pooling $ {\mathbf{N}}^{4} = {\mu }_{2}\left( {{\mathbf{N}}^{3},{\mathbf{A}}^{\prime \prime }}\right) $ $ {\mathbf{N}}^{4} = \left\lbrack  \begin{matrix} {\mathbf{N}}^{4} \\  {\mathbf{0}}^{\mathrm{b}} \end{matrix}\right\rbrack $ $ {\mathbf{N}}^{4} = {\mathbf{N}}^{4} + {\mathbf{N}}_{1}^{1} $ ; skip-connection</td><td>$ {\mathbf{N}}^{4} $</td></tr><tr><td>2.</td><td>$ {\mathbf{N}}^{1} $ $ {\mathbf{A}}^{\prime \prime } $</td><td>pooling $ {\mathbf{N}}_{1}^{2} = {\mu }_{1}\left( {{\mathbf{N}}^{1},{\mathbf{A}}^{\prime \prime }}\right) $ $ {\mathbf{N}}_{2}^{2} = {\mu }_{1}\left( {{\mathbf{N}}^{1},{\mathbf{A}}^{\prime \prime }}\right) $ $ {\mathbf{N}}_{2}^{2} \odot  {\mathbf{N}}_{1}^{2} $ $ {\mathbf{N}}^{2} = {\mathbf{N}}_{1}^{2} + {\mathbf{N}}_{2}^{2} $ Remove entries of $ {\mathbf{N}}^{\mathrm{c}} $ from $ {\mathbf{N}}^{2} $</td><td>$ {\mathrm{N}}^{2} $ $ {\mathrm{N}}_{1}^{2} $</td><td>5. 6.</td><td>$ {\mathrm{N}}^{4} $ $ {\mathbf{A}}^{\prime } $ $ {\mathbf{N}}_{1}^{0} $ $ {\mathbf{N}}^{5} $ A</td><td>un-pooling $ {\mathbf{N}}^{5} = {\mu }_{2}\left( {{\mathbf{N}}^{4},{\mathbf{A}}^{\prime }}\right) $ $ {\mathbf{N}}^{5} = \left\lbrack  \begin{matrix} {\mathbf{N}}^{5} \\  {\mathbf{0}}^{\mathrm{r}} \end{matrix}\right\rbrack $ $ {\mathbf{N}}^{5} = {\mathbf{N}}^{5} + {\mathbf{N}}_{1}^{0} $ ; skip-connection GCN $ {\mathbf{N}}^{6} = {\mu }_{2}\left( {{\mathbf{N}}^{5},\mathbf{A}}\right) $</td><td>$ {\mathrm{N}}^{5} $ $ {\mathbf{N}}^{\prime } $</td></tr></table>

![0196c354-f366-7182-8e00-4fe98707e99c_5_105_154_1543_840_0.jpg](images/0196c354-f366-7182-8e00-4fe98707e99c_5_105_154_1543_840_0.jpg)

Fig. 3. Computational graph of HEBG.

![0196c354-f366-7182-8e00-4fe98707e99c_5_206_1086_1331_503_0.jpg](images/0196c354-f366-7182-8e00-4fe98707e99c_5_206_1086_1331_503_0.jpg)

Fig. 4. Proposed MDP for optimization.

Unlike the policy function, the value function computes the accumulated reward $ Q $ which is a scalar value. Note that when $ {h}^{\prime } > 1 $ , the output of the HEBG denoted as $ {\mathbf{N}}^{\prime } $ is a matrix as defined in Eq. (11). Therefore, a global sum pooling operation (Aich and Stavness, 2019) is applied to convert $ {\mathbf{N}}^{\prime } $ into a vector before sending it to the multi-layer perceptrons (MLPs) that take vectors as inputs to compute the output Q. The agent takes state and action as inputs to compute the value function as follows:

$$
\mathrm{Q}\left( {\mathbf{N},{\widetilde{\mathbf{A}}}_{\mathrm{n}},\mathbf{\pi }}\right)  = \acute{\mathrm{Q}} \in  {\mathbb{R}}^{1 \times  1} \tag{13a}
$$

$$
\acute{Q} = \operatorname{MLP}\left( {\operatorname{GSP}\left( {\operatorname{HEBG}\left( {\mathbf{N},\mathbf{A}}\right) }\right) \parallel \operatorname{GSP}\left( {\operatorname{HEBG}\left( {\mathbf{\pi },\mathbf{A}}\right) }\right) }\right)  \tag{13b}
$$

where $ \operatorname{GSP}\left( \mathbf{N}\right)  = \left\lbrack  \begin{array}{lll} \mathop{\sum }\limits_{{i = 1}}^{n}{\mathrm{n}}_{i,1} & \cdots & \mathop{\sum }\limits_{{i = 1}}^{n}{\mathrm{n}}_{i, g} \end{array}\right\rbrack   \in  {\mathbb{R}}^{1 \times  g},\;\left( {\mathrm{p}\parallel \mathrm{q}}\right) $ , and $ \operatorname{MLP}\left( \cdot \right) $ denote the global sum pooling operation, concatenation operation, and MLP function, respectively.

### 5.3. Action

The structure is modified by the output of the policy function $ \acute{\mathbf{\pi }} \in  {\mathbb{R}}^{\left( {{f}_{\mathrm{f}} + {c}_{\mathrm{f}} + {b}_{\mathrm{f}} + {r}_{\mathrm{f}}}\right)  \times  1} $ . In this research, the modification (i.e., an action in RL) is to add a brace or increase the sectional index of a structural element. The irrelevant elements for which modifications cannot be made, such as adding a brace into the region that already has a brace, or increasing the size of an element that already has the largest index, are removed from the target of action by assigning 0 at the corresponding component in $ \acute{\pi } $ .

Each value $ {\widehat{\pi }}_{j} $ in $ \widehat{\mathbf{\pi }} $ is also multiplied by a weight $ {w}_{j} $ which indicates the importance of the element to be modified, e.g., the beam or column elements that violate the long-term stress constraint may be more important than those that do not violate any constraint. Let $ {\sigma }_{j}^{\mathrm{L}, t} $ and $ {\sigma }_{j}^{\mathrm{S}, t} $ be the long-term and short-term stress ratios of an element $ j $ in the MDP step $ t $ . The value of $ {w}_{j} $ is assigned as

$$
{w}_{j} = \left\{  \begin{matrix} {1.0} & \left( {j\text{ violates the stress constraints }\left( {{\sigma }_{{j}^{1}}^{\mathrm{L}, t} > 1}\right)  \cup  \left( {{\sigma }_{{j}^{1}}^{\mathrm{S}, t} > 1}\right) }\right) \\  {0.9} & \left( {j\text{ is a beam or column satisfying }\left( {{\sigma }_{{j}^{1}}^{\mathrm{L}, t} \leq  1}\right)  \cap  \left( {{\sigma }_{{j}^{1}}^{\mathrm{S}, t} \leq  1}\right) }\right) \\  {0.9} & \left( {j\text{ is an added brace satisfying }\left( {{\sigma }_{{j}^{1}}^{\mathrm{L}, t} \leq  1}\right)  \cap  \left( {{\sigma }_{{j}^{1}}^{\mathrm{S}, t} \leq  1}\right) }\right) \\  {0.95} & \left( {j\text{ is a brace that can be added into the structure}}\right)  \end{matrix}\right.
$$

(14)

Table 2

Values in the row $ i $ of the node feature matrix $ \mathbf{N} $

<table><tr><td>$ {n}_{\mathrm{i},1} $</td><td>1 if the node is the floor else 0</td><td>$ {n}_{\mathrm{i},{28}} $</td><td>$ x $ -coordinate of the start node</td></tr><tr><td>$ {n}_{i,2} $</td><td>$ {n}_{i,1} \times $ floor height</td><td>$ {n}_{i,{29}} $</td><td>$ y $ -coordinate of the start node</td></tr><tr><td>$ {n}_{i,3} $</td><td>1 if the element is column with largest sectional index else 0</td><td>$ {n}_{i,{30}} $</td><td>$ z $ -coordinate of the start node</td></tr><tr><td>$ {n}_{i,4} $</td><td>1 if the element is column else 0</td><td>$ {n}_{i,{31}} $</td><td>1 if start node has support else 0</td></tr><tr><td>$ {n}_{i,5} $</td><td>1 if the element is beam with largest sectional index else 0</td><td>$ {n}_{i,{32}} $</td><td>1 if start node has no support else 0</td></tr><tr><td>$ {n}_{i,6} $</td><td>1 if the element is beam else 0</td><td>$ {n}_{i,{33}} $</td><td>$ z $ -directional long-term load applied to the start node</td></tr><tr><td>$ {n}_{i,7} $</td><td>1 if the element is brace with largest sectional index else 0</td><td>$ {n}_{i,{34}} $</td><td>$ x $ -directional short-term load applied to the start node</td></tr><tr><td>$ {n}_{i,8} $</td><td>1 if the element is brace else 0</td><td>$ {n}_{i,{35}} $</td><td>$ y $ -directional short-term load applied to the start node</td></tr><tr><td>$ {n}_{i,9} $</td><td>1 if the element is brace with smallest sectional index else 0</td><td>$ {n}_{i,{36}} $</td><td>1 if start node is the center node of the beam in $ x $ -direction else 0</td></tr><tr><td>$ {n}_{i,{10}} $</td><td>Sectional area</td><td>$ {n}_{i,{37}} $</td><td>1 if start node is the center node of the beam in $ y $ -direction else 0</td></tr><tr><td>$ {n}_{i,{11}} $</td><td>Second moment of area along local $ {v}_{1} $ -axis $ \left( {I}_{1}\right) $</td><td>$ {n}_{i,{38}} $</td><td>$ x $ -directional displacement of the start node</td></tr><tr><td>$ {n}_{i,{12}} $</td><td>Second moment of area along local $ {v}_{2} $ -axis $ \left( {I}_{2}\right) $</td><td>$ {n}_{i,{39}} $</td><td>$ y $ -directional displacement of the start node</td></tr><tr><td>$ {n}_{i,{13}} $</td><td>Length</td><td>$ {n}_{i,{40}} $</td><td>$ z $ -directional displacement of the start node</td></tr><tr><td>$ {n}_{i,{14}} $</td><td>$ {\sigma }_{i}^{\mathrm{L}} $</td><td>$ {n}_{i,{41}} $</td><td>$ x $ -coordinate of the end node</td></tr><tr><td>$ {n}_{i,{15}} $</td><td>$ {\sigma }_{i}^{S} $ : positive $ x $ -directional load</td><td>$ {n}_{i,{42}} $</td><td>$ y $ -coordinate of the end node</td></tr><tr><td>$ {n}_{i,{16}} $</td><td>$ {\sigma }_{i}^{\mathrm{S}} $ : negative $ x $ -directional load</td><td>$ {n}_{i,{43}} $</td><td>$ z $ -coordinate of the end node</td></tr><tr><td>$ {n}_{i,{17}} $</td><td>$ {\sigma }_{i}^{S} $ : positive $ y $ -directional load</td><td>$ {n}_{i,{44}} $</td><td>1 if end node has support else 0</td></tr><tr><td>$ {n}_{i,{18}} $</td><td>$ {\sigma }_{i}^{\mathrm{S}} $ : negative $ y $ -directional load</td><td>$ {n}_{i,{45}} $</td><td>1 if end node has no support else 0</td></tr><tr><td>$ {n}_{i,{19}} $</td><td>$ {\delta }_{i} $</td><td>$ {n}_{i,{46}} $</td><td>$ z $ -directional long-term load applied to the end node</td></tr><tr><td>$ {n}_{i,{20}} $</td><td>$ {\theta }_{i} $</td><td>$ {n}_{i,{47}} $</td><td>$ x $ -directional short-term load applied to the end node</td></tr><tr><td>$ {n}_{i,{21}} $</td><td>1 if $ {n}_{i,{14}} > 1 $ else 0</td><td>$ {n}_{i,{48}} $</td><td>$ y $ -directional short-term load applied to the end node</td></tr><tr><td>$ {n}_{i,{22}} $</td><td>1 if $ {n}_{i,{15}} > 1 $ else 0</td><td>$ {n}_{i,{49}} $</td><td>1 if end node is the middle node on the beam lies along $ \mathrm{x} $ -axis else 0</td></tr><tr><td>$ {n}_{i,{23}} $</td><td>1 if $ {n}_{i,{16}} > 1 $ else 0</td><td>$ {n}_{i,{50}} $</td><td>1 if end node is the middle node on the beam lies along y-axis else 0</td></tr><tr><td>$ {n}_{i,{24}} $</td><td>1 if $ {n}_{i,{17}} > 1 $ else 0</td><td>$ {n}_{i,{51}} $</td><td>$ x $ -directional deformation of the end node</td></tr><tr><td>$ {n}_{i,{25}} $</td><td>1 if $ {n}_{i,{18}} > 1 $ else 0</td><td>$ {n}_{i,{52}} $</td><td>$ y $ -directional deformation of the end node</td></tr><tr><td>$ {n}_{i,{26}} $</td><td>1 if $ {n}_{i,{19}} > 1 $ else 0</td><td>$ {n}_{i,{53}} $</td><td>$ z $ -directional deformation of the end node</td></tr><tr><td>$ {n}_{i,{27}} $</td><td>1 if $ {n}_{i,{20}} > 1 $ else 0</td><td/><td/></tr></table>

and $ {\acute{\pi }}_{j} $ becomes

$$
{\acute{\pi }}_{j} \leftarrow  {\acute{\pi }}_{j} \times  {w}_{j} \tag{15}
$$

Finally, the modification is applied to the element $ i = \operatorname{argmax}\left( {\widehat{\pi }}_{i}\right) $ excluding irrelevant elements and prioritizing the elements that violate constraints. After the element is modified, the following three rules, derived from structural requirements, are further applied.

Table 3

Values in the adjacency matrix $ \mathbf{A} $

<table><tr><td>$ {a}_{{ni}, j} $</td><td>1 if there is a connection between nodes $ i $ and $ j $ , else 0</td></tr></table>

Rule 1: The sectional index of a column should not be larger than the one below it. This rule is enforced by increasing the sectional indices of the lower columns that are smaller than those above them.

Rule 2: The beams on the same floor should have the same sectional index. This rule is imposed by determining the value of the largest beam sectional index on the floor and assigning it to all beam sectional indices on that floor.

Rule 3: The brace cannot be added on the first floor. This rule ensures the functionality of the structure.

After the agent modifies the structure and these rules are applied, the objective and constraint functions are evaluated from structural analysis for computing the reward.

### 5.4. Reward

The reward is computed from the change in objective function $ {\Delta V} $ and all the changes in constraint function values to evaluate the increase of volume and the improvement in terms of the constraints. Let a $ p $ -norm of a vector $ \mathbf{x} \in  {\mathbb{R}}^{{n}_{x} \times  1} $ be $ \parallel \mathbf{x}{\parallel }_{p} = \sqrt[p]{\mathop{\sum }\limits_{{i = 1}}^{{n}_{x}}{\left( {x}_{i}\right) }^{p}} $ . Reward $ {\mathrm{R}}_{t + 1} $ given to the agent after making action $ {\mathrm{A}}_{t} $ is computed as:

$$
{\mathrm{R}}_{t + 1} =  - {\Delta V} \times  \left\lbrack  {4 - \left( {\Delta {c}_{1} + \Delta {c}_{2} + \Delta {c}_{3} + \Delta {c}_{4}}\right) }\right\rbrack   \tag{16a}
$$

$$
{\Delta V} = {V}_{t + 1} - {V}_{t} \tag{16b}
$$

$$
\Delta {c}_{1} = \left\{  \begin{matrix} \frac{{\begin{Vmatrix}{\mathbf{\sigma }}^{\mathrm{L}}\end{Vmatrix}}_{p}^{t} - {\begin{Vmatrix}{\mathbf{\sigma }}^{\mathrm{L}}\end{Vmatrix}}_{p}^{t + 1}}{{\begin{Vmatrix}{\mathbf{\sigma }}^{\mathrm{L}}\end{Vmatrix}}_{p}^{t + 1} + 1} & \left( {\text{ if }{\sigma }_{\max }^{\mathrm{L}, t + 1} > 1}\right) \\  1 & \text{ (else) } \end{matrix}\right.  \tag{16c}
$$

$$
\Delta {c}_{2} = \left\{  \begin{matrix} \frac{{\begin{Vmatrix}{\mathbf{\sigma }}^{\mathrm{S}}\end{Vmatrix}}_{p}^{t} - {\begin{Vmatrix}{\mathbf{\sigma }}^{\mathrm{S}}\end{Vmatrix}}_{p}^{t + 1}}{{\begin{Vmatrix}{\mathbf{\sigma }}^{\mathrm{S}}\end{Vmatrix}}_{p}^{t + 1} + 1} & \left( {\text{ if }{\sigma }_{\max }^{\mathrm{S}, t + 1} > 1}\right) \\  1 & \text{ (else) } \end{matrix}\right.  \tag{16d}
$$

$$
\Delta {c}_{3} = \left\{  \begin{matrix} \frac{{\theta }_{\max }^{t} - {\theta }_{\max }^{t + 1}}{{\theta }_{\max }^{t + 1} + 1} & \left( {\text{ if }{\theta }_{\max }^{t + 1} > 1}\right) \\  1 & \text{ (else) } \end{matrix}\right.  \tag{16e}
$$

$$
\Delta {c}_{4} = \left\{  \begin{matrix} \frac{{\delta }_{\max }^{t} - {\delta }_{\max }^{t + 1}}{{\delta }_{\max }^{t + 1} + 1} & \left( {\text{ if }{\delta }_{\max }^{t + 1} > 1}\right) \\  1 & \text{ (else) } \end{matrix}\right.  \tag{16f}
$$

where $ {V}_{t + 1} $ and $ {V}_{t} $ are values of the objective function after and before modification, respectively. $ {\begin{Vmatrix}{\mathbf{\sigma }}^{\mathrm{L}}\end{Vmatrix}}_{p}^{t},{\begin{Vmatrix}{\mathbf{\sigma }}^{\mathrm{L}}\end{Vmatrix}}_{p}^{t + 1} $ , and $ {\sigma }_{\max }^{\mathrm{L}, t + 1} $ denote the $ p $ -norm of the long-term stress ratio before modification, the $ p $ -norm of the long-term stress ratio after modification, and the maximum long-term stress ratio after modification, respectively, where $ p = {10} $ in the numerical examples. The values corresponding to $ {\begin{Vmatrix}{\mathbf{\sigma }}^{\mathrm{S}}\end{Vmatrix}}_{p}^{t},{\begin{Vmatrix}{\mathbf{\sigma }}^{\mathrm{S}}\end{Vmatrix}}_{p}^{t + 1},{\sigma }_{\max }^{\mathrm{S}, t + 1},{\theta }_{\max }^{t + 1},{\theta }_{\max }^{t} $ , $ {\delta }_{\max }^{t + 1} $ , and $ {\delta }_{\max }^{t} $ are defined similarly.

### 5.5. Optimization algorithm

At the beginning of the optimization process, the beams, columns, and braces are initialized to the smallest section indices as shown in Fig. 5A. Note that the smallest section index of the brace indicates the absence of the brace. Braces can be assigned at the outer structural grids or perimeter so that the braces do not interfere with the room arrangement in the building. Fig. 5B shows four brace types that can be added into the frame: L-brace, R-brace, V-brace, and K-brace. L-brace and R-brace are classified when viewed from the exterior of the building. Fig. 6 illustrates the optimization algorithm in this research.

![0196c354-f366-7182-8e00-4fe98707e99c_7_360_153_1025_534_0.jpg](images/0196c354-f366-7182-8e00-4fe98707e99c_7_360_153_1025_534_0.jpg)

Fig. 5. Structure with braces: (A) initial frame, (B) brace types.

![0196c354-f366-7182-8e00-4fe98707e99c_7_362_782_1026_599_0.jpg](images/0196c354-f366-7182-8e00-4fe98707e99c_7_362_782_1026_599_0.jpg)

Fig. 6. Optimization algorithm.

Table 4

List of sectional properties of column.

<table><tr><td>Index</td><td>Square HSS (H-H-t)</td><td>$ A\left( {\mathrm{\;{cm}}}^{2}\right) $</td><td>$ I\left( {\mathrm{\;{cm}}}^{4}\right) $</td><td>$ Z\left( {\mathrm{\;{cm}}}^{3}\right) $</td></tr><tr><td>0</td><td>$ {200} \times  {200} \times  9 $</td><td>66</td><td>3920</td><td>392</td></tr><tr><td>1</td><td>$ {300} \times  {300} \times  {12} $</td><td>133</td><td>18100</td><td>1200</td></tr><tr><td>2</td><td>$ {400} \times  {400} \times  {16} $</td><td>237</td><td>57100</td><td>2850</td></tr><tr><td>3</td><td>$ {500} \times  {500} \times  {22} $</td><td>404</td><td>150000</td><td>6010</td></tr><tr><td>4</td><td>$ {600} \times  {600} \times  {32} $</td><td>727</td><td>392000</td><td>13100</td></tr><tr><td>5</td><td>$ {700} \times  {700} \times  {35} $</td><td>855</td><td>637000</td><td>18210</td></tr><tr><td>6</td><td>$ {800} \times  {800} \times  {40} $</td><td>1216</td><td>1173000</td><td>29360</td></tr></table>
