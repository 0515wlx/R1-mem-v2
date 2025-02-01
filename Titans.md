# Titans: Learning to Memorize at Test Time

**Ali Behrouz\*, Peilin Zhong\*, and Vahab Mirrokni\***

Google Research
{alibehrouz, peilinz, mirrokni}@google.com

## Abstract

Over more than a decade there has been an extensive research effort of how effectively utilize recurrent models and attentions. While recurrent models aim to compress the data into a fixed-size memory (called hidden state), attention allows attending to the entire context window, capturing the direct dependencies of all tokens. This more accurate modeling of dependencies, however, comes with a quadratic cost, limiting the model to a fixed-length context. We present a new neural long-term memory module that learns to memorize historical context and helps an attention to attend to the current context while utilizing long past information. We show that this neural memory has the advantage of a fast parallelizable training while maintaining a fast inference. From a memory perspective, we argue that attention due to its limited context but accurate dependency modeling performs as a short-term memory, while neural memory due to its ability to memorize the data, acts as a long-term, more persistent, memory. Based on these two modules, we introduce a new family of architectures, called Titans, and present three variants to address how one can effectively incorporate memory into this architecture. Our experimental results on language modeling, commonsense reasoning, genomics, and time series tasks show that Titans are more effective than Transformers and recent modern linear recurrent models. They further can effectively scale to larger than 2M context window size with higher accuracy in needle-in-haystack tasks compared to baselines.

## 1 Introduction

Transformers, pure attention-based architectures (Vaswani et al. 2017), have been firmly established as state-of-the-art models in sequence modeling, mainly due to their in-context learning and ability to learn at scale (Kaplan et al. 2020). The primary building blocks of Transformers—attention modules—function as associative memory blocks (Bietti et al. 2024), where they learn to store key-value associations and retrieve them by computing pairwise similarity between queries (i.e., search signals) and keys (i.e., contexts). Accordingly, by design, the output of a Transformer is exclusively conditioned on the direct dependencies of tokens in the current context window. This accurate modeling of dependencies, however, comes with quadratic time and memory complexity in terms of the context length. In complex real-world tasks (e.g., language modeling (N. F. Liu et al. 2024), video understanding (C.-Y. Wu et al. 2019), long-term time series forecasting (H. Zhou et al. 2021)), the context window can become extremely large, making the applicability of Transformers challenging in these downstream tasks.

To overcome the scalability issue of Transformers, recent studies aim to design different variants of linear Transformers (Kacham, Mirrokni, and P. Zhong 2024; Katharopoulos et al. 2020; S. Yang, B. Wang, Shen, et al. 2024), where softmax is replaced by a kernel function in the attention (see §2.1 for details), resulting in a significant drop in memory consumption. Despite efficiency and the ability to scale to longer context, linear Transformers do not show competitive performance compared to Transformers as the kernel trick makes the model a linear recurrent network, in which the data is compressed into a matrix-valued states (Katharopoulos et al. 2020). This, however, brings a contradictory fact about linear recurrent (or linear Transformers) models: On one hand, we use these linear models to enhance scalability and efficiency (linear vs. quadratic complexity), whose advantages is appeared for very long context; On the other hand, a very long context cannot be properly compressed in a small vector-valued or matrix-valued states (S. Wang 2024).

Furthermore, beyond efficiency, most existing architectures—ranging from Hopfield Networks (Hopfield 1982) to LSTMs (Jürgen Schmidhuber and Hochreiter 1997) and Transformers (Vaswani et al. 2017)—face challenges when dealing with generalization, length extrapolation, and/or reasoning (An et al. 2022; Qin, Y. Zhong, and Dong 2024), all of which are inseparable parts of many hard real-world tasks. Although these architectures draw inspiration from the human brain, each of which are missing: (1) a crucial component for learning (Powers)—such as short-term memory, long-term memory, meta-memory, attending to current context, etc. (Cowan 2008); (2) how these components are interconnected systems that can operate independently; and/or (3) the ability to actively learn from data and memorize the abstraction of past history. We argue that in an effective learning paradigm, similar to human brain, there are distinct yet interconnected modules, each of which is responsible for a component crucial to the learning process.

### Memory Perspective
Memory is a fundamental mental process and is an inseparable component of human learning (Terry 2017). Without a properly functioning memory system, humans and animals would be restricted to basic reflexes and stereotyped behaviors. Accordingly, memory has been the inspiration for many seminal research in machine learning literature; e.g., Hopfield Networks (Hopfield 1982), LSTMs (Jürgen Schmidhuber and Hochreiter 1997), and Transformers (Vaswani et al. 2017).

Taking inspiration from the common definitions of memory and learning in neuropsychology literature (Ozano, Hirano, and Balaban 2000), most existing architectures consider memory as a neural update caused by an input, and define learning as a process for acquiring effective and useful memory, given an objective. In this perspective, Recurrent Neural Networks (RNNs) (Williams and Zipser 1989) can be defined as models with a vector-valued memory module M (also called hidden state) with two main steps: Given a new input x_t at time t, the model (1) updates the memory using a function F(M_{t-1}, x_t) (with compression); and (2) retrieves the corresponding memory of input using a function of M_k (use §2.1 for details). Similarly, Transformers can be seen as architectures with a growing memory and two similar steps. That is, in pair of key and value matrices acts as the model's memory, and the model: (1) updates the memory by appending the key and value to current memory (uncompressed), and (2) retrieves key vectors from memory by finding the similarity of query and key vectors, which is then used to weight the values for the output.

This perspective can help us better understand existing paradigms, their critical differences, and design more effective architectures. For example, the main difference between Transformers (Vaswani et al. 2017) and linear Transformers (Katharopoulos et al. 2020) is the memory structure as well as the memory updating step, in which linear Transformers compress the historical data into a fixed-size matrix-valued memory while Transformers keep all historical data (within the context length) without any compression. While both linear Transformers and linear RNNs (including state space models) compress the information in memory update step, the critical difference lies in the structure of the memory, where linear RNNs (vs. linear Transformers) use a vector-valued memory (vs. matrix-valued memory). Therefore, this perspective motivates us to ask: (Q1) What constitute a good structure for the memory? (Q2) What is a proper memory update mechanism? and (Q3) What is a good memory retrieval process?

Revisiting our understanding of human memory, it is neither a unitary process nor it serves a single function (Cowan 2008). In fact, memory is a collection of systems—e.g., short-term, working, and long-term memory—each serving a different function with different neural structures, and each capable of operating independently (Wiilingham 1997). This fact motivates us to ask: (Q4) How to design an efficient architecture that incorporates different interconnected memory modules? Finally, storing a memory is a neural process that requires to encode and store the abstraction of the past. It can be over-simplification to assume a single vector or a matrix, whose parameters are encoding the data in a linear manner, be enough for storing long-term history. (Q5) Is a deep memory module needed to effectively store/remember long past?

### Contributions and Roadmap
In this paper, we aim to answer the above five questions by designing a long-term neural memory module, that can efficiently and effectively learn to memorize at test time. Building upon its design, we discuss how it can be incorporated into an architecture.

### Neural Memory (§3)
We present a (deep) neural long-term memory that (as a meta-in-context model) learns how to memorize/store the data into its parameters at test time. Inspired by human long-term memory system (Mandler 2014),
we design this memory module so an event that violates the expectations (being surprising) is more memorable. To this end, we measure the surprise of an input with the gradient of the neural network with respect to the input in associative memory loss (§3.1 for details). To better handle the limited memory, we present a decaying mechanism that consider the proportion of memory size and the amount of data surprise, resulting in better memory management. We show that this decay mechanism is in fact the generalization of forgetting mechanism in modern recurrent models (Dao and Gu 2024; Gu and Dao 2024; S. Yang, Kautz, and Hatamizadeh 2024). Interestingly, we find that this mechanism is equivalent to optimizing a meta neural network with mini-batch gradient descent, momentum, and weight decay. Building upon tensorizing mini-batch gradient descent to use more matmul operations (Yu Sun et al. 2024), we present a fast and parallelizable algorithm to train our deeper long-term memory.

Titans Architectures (§4). After designing the long-term neural memory, an important remaining question is how to effectively and efficiently incorporate memory into a deep learning architecture. We present Titans, a family of deep models that consists of three hyper-heads: (1) Core: this module consists of the short-term memory, and is responsible for the main flow of processing the data (we use attention with limited window size); (2) Long-Term Memory: this branch is our neural long-term memory module that is responsible to store/remember long past; (3) Persistent Memory: this is a set of learnable but date-independent parameters that encodes the knowledge about a task. Finally, as a proof of concept, we present three variants of Titans, in which we incorporate memory as: (i) a context, (ii) a layer, and (iii) a gated branch.

Experimental Results (§5). We perform experimental evaluations on language modeling, commonsense reasoning, recall-intense, needle in haystack, time series forecasting, and DNA modeling tasks. We observe that our Titan architecture outperforms all modern recurrent models as well as their hybrid variants (combining with sliding-window attention) across a comprehensive set of benchmarks. Furthermore, Titans outperforms Transformers with the same context window, and show competitive performance with Transformers that use the entire context. These results are achieved while, contrary to Transformers, Titans scale to larger than 2M context window size.

2 Preliminaries

In this section, we discuss the notation and some background concepts that we use though the paper. We let x ∈ RN×d be the input, M be a neural network (neural memory module), Q, K, V be the query, key and value of the attention mechanism, and M be the attention mask. When segmenting the sequence, we use s(i) to refer to the i-th segment. Through the paper, we abuse the notation and use subscripts to refer to a specific element of a matrix, vector, or segments. For example, we let s(j) be the j-th token in the i-th segment. The only exception is subscripts with t, which we reserved to index recurrence over time, or the state of a neural network at time t. Given a neural network N and a data sample x, we use N(x) to refer to the forward pass with (resp. without) weight adjustment. Also, we abuse the notation and use N(k) to refer to the k-th layer of the neural network. In the following, we first discuss the backgrounds for attention and its different variants followed by a review of modern linear RNNs. Finally, we discuss a memory perspective of these architectures that motivates us to design Titans.

2.1 Backgrounds

Attention. Transformers (Vaswani et al. 2017) as the de facto backbone for many deep learning models are based on attention mechanism. Given input x ∈ RN×d, causal attention computes output y ∈ RN×d on softmax over input dependent key, value, and query matrices:

Q = xWQ, K = xWK, V = xWV,
(1)
yi = ∑ j=1i exp(QT i Kj/ √ d)vj
∑ l∈1i exp(QT i Kl/ √ d)
(2)

where WQ, WK, and WV ∈ Rd×d are learnable parameters. Despite the power and effectiveness in recall, transformers need at least N × d operators to calculate the output, resulting in larger memory consumption and lower-throughput for longer sequences.

Efficient Attentions. To improve the memory consumption and throughput of softmax attention for longer sequences, various studies focused on I/O aware implementations of attention (Dao 2024; Dao, D. Fu, et al. 2022), designing more
efficient attention mechanisms by sparsifying the attention matrix (B. Chen et al. 2021; Choromanski et al. 2021; Dai et al. 2019), approximating the softmax (Arora et al. 2024), or developing kernel-based (linear) attentions (Aksenov et al. 2024; Kacham, Mirochnick, and P. Zhong 2024; Schlag et al. 2021; Irie, and Jürgen Schmidhuber 2021; S. Yang, B. Wang, Shen, et al. 2024). In this part, we focus on the latter, i.e., linear attentions, where the softmax in standard attention is replaced with an alternative kernel function φ(·, ·), such that φ(x, y) = φ(x)φ(y). Accordingly, the attention can be written as:

$$y_i = \frac{1}{\sum_{j=1}^{l} \phi(Q^{T}K_{j})} \sum_{j=1}^{l} \phi(Q^{T}K_{j})V_{j} = \frac{\phi(Q^{T}) \sum_{j=1}^{l} \phi(K_{j})V_{j}}{\phi(Q^{T}) \sum_{i=1}^{l} \phi(K_{i})}$$

resulting in a higher-throughput as terms ∑_{j=1}^{l} φ(K_{j}) and ∑_{i=1}^{l} φ(K_{i}) are re-using in each step. When choosing the kernel as identity matrix (Yutao Sun et al. 2023), the above formulation can also be written in a recurrent format:

$$M_{t} = M_{t-1} + K_{t} V_{t},$$

$$y_{t} = Q_{t} M_{t},$$

which allows efficient inference for linear attentions.

Modern Linear Models And Their Memory Perspective. As discussed earlier, one can define learning as a process for acquiring effective and useful memory. Building upon this, one can see the hidden state of Recurrent Neural Networks (RNNs) as a memory unit, which the model uses to compress the information into. Accordingly, in a general form of recurrent neural network, the hidden state can be treated as a memory unit and requires can be split into the read and write operations in the memory unit. That is, let x ∈ Rnxd be the memory unit, and y ∈ Rd be the output, then the general form of the recurrent neural network is defined as:

$$M_{t} = f(M_{t-1}, x_{t}),$$

$$y_{t} = g(M_{t}, x_{t}),$$

where f(·) is the read and g(·) is the write corresponding function. Note that here the subscript of M_{t} shows the state of the memory at time t.

In this perspective, the recurrence formula of linear Transformers (see Equation 4) is equivalent to additively compress each input keys and values, (K_{t}, V_{t}), into a matrix-valued memory M_{t}. Therefore, when dealing with long context data, this additive nature of the process results in memory overflow, significantly damaging the performance of the model. To address this, studies have focused on two prominent directions: (1) Adding memory, Shen et al. 2024, LRU (Overett et al. 2023), Griffin (De et al. 2021), xLSTM (Beck et al. 2024), and Mamba2 (Dao and Gu 2024), which the later is also connected to the discretized version of traditional state space models (Gua and Dao 2024) 2) Improving the write operation: To overcome the additive nature of memory write operation in traditional recurrent models, Widrow and Hoff (1988) presented Delta Rule, in which before adding a memory (i.e., a pair of key and value), the model first removes its past value. To enhance the parallelizable training and scaling, S. Yang, B. Wang, Yu Zhang, et al. (2024) present a fast parallelizable algorithm. Finally, very recently, S. Yang, Kautz, and Hatamizadeh (2024) improved the DeltaNets by adding a forget gate.

Memory Modules. Memory has always been one of the core parts of the neural network designs (Graves, Wayne, and Danihelka 2014; JH Schmidhuber 1992; Jürgen Schmidhuber and Hochreiter 1997; J. Zhang et al. 2024). The idea of seeing linear layers as the key-value (associative) memory system backs to fast weight programs, in which dynamic fast programs are incorporated into recurrent neural networks to serve as writable memory (JH Schmidhuber 1992). The learning rules of Hebbian (Hebb 2005) and delta (Prados and Kak 1989) are the most popular learning rules for fast weight programs, which have been extensively explored in various studies (Irie, Schlag, et al. 2021; Munkhdalai, Sordoni, et al. 2019; Munkhdalai and H. Yu 2017; Schlag, Irie, and Jürgen Schmidhuber 2021; JH Schmidhuber 1992; S. Yang, Kautz, and Hatamizadeh 2024; S. Yang, B. Wang, Yu Zhang, et al. 2024). All these models, however, are based on memory surprise, missing the token flow in the sequences (see Section 3.1), and most of them lacks a forgetting gate, resulting in a poor memory management.

We further discuss the connection of our architectures with recent models in Appendix C. Additional related work are discussed in Appendix A.
# 3 Learning to Memorize at Test Time

To overcome the lack of long-term memory and to enable the model to learn, forget, and retrieve information, in this section, we present a neural long-term memory module, which is a meta model that learns to memorize at test time. In Section 3.1, we first discuss the motivation and the design of the neural memory. In Section 3.2, we discuss how our architecture design can benefit from a fast and parallelizable training. Finally, in Section 3.3, we augment our architecture using persistent memory module, in which we use learnable but data-independent parameters to learn meta information about the task.

## 3.1 Long-term Memory

To design a neural long-term memory module, we need a model that can encode the abstraction of the past history into its parameters. An example of this can be LLMs that are shown to be memorizing their training data (Leybzon and Kervadec 2024; Schwarzschild et al. 2024; Staab et al. 2024). Therefore, a simple idea is to train a neural network and expect it to memorize its training data. Memorization, however, has almost always been known as an undesirable phenomena in neural networks as it limits the model generalization (Bayat et al. 2024), and it causes privacy concerns (Staab et al. 2024), and so results in poor performance at test time. Moreover, the memorization of the training data might not be helpful at test time, in which the data might be out-of-distribution. We argue that, instead, we need an online meta-model that learns how to memorize/forget the data at test time. In this setup, the model is learning a function that is capable of memorization, but it is not overwriting to the training data, resulting in a better generalization at test time.

**Learning Process and Surprise Metric.** The key idea to train a long-term memory is to treat its training as an online learning problem, in which we aim to compress the past information \(x_1, \ldots, x_{t-1}\) into the parameters of our long-term neural memory module \(M_t\). As discussed earlier, we argue that violates the expectations (i.e., is surprising) is more memorable for humans (Mandler 2014). Inspired by this, a simple definition of surprise for a model can be its gradient with respect to the memory. The larger the gradient is, the more different the input data is from the past data. Accordingly, using this surprise score, we can update the memory as:

\[
M_t = M_{t-1} - \theta_t \nabla \mathcal{L}(M_{t-1:x_t})
\]

This surprise metric, however, can result in missing important information that comes after a big surprising moment. That is, the gradient can become extremely small after several surprising steps, leading to stocking in a flat area (i.e., local minima), and missing information about some parts of the sequence. From the human memory perspective, an event might not consistently surprise us through a long period of time although it is memorable. The reason is that the initial moment is surprising enough to get our attention through a long time frame, leading to memorizing the entire time frame. To improve the above surprise metric (Equation 8), we break the surprise metric into (1) past surprise, which measures the surprise amount of a very recent past; and (2) momentary surprise, which measures the surprise of incoming data:

\[
M_t = M_{t-1} + S_t
\]

\[
S_t = \eta_t S_{t-1} - \theta_t \nabla \mathcal{L} (M_{t-1:x_t}).
\]

Interestingly, this formulation is similar to gradient descent with momentum, where \(S_t\) is the momentum element. Therefore, the momentum here acts as a memory of surprise across time (sequence length). In this formulation, the term \(\eta_t\) is a data-dependent surprise decay (a function of \(x_t\)), controlling how our surprise decays over time, and the term \(\theta_t\) is controlling how much of momentary surprise should be incorporated into the final surprise metric in a data-dependent manner. This data-dependency is particularly important in this design: While surprise of previous tokens might be needed to affect the surprise of the next token, it is mostly valid if all tokens are relevant and are in the same context. Accordingly, a data-dependent \(n\) can control if memory needs to: (1) ignore the last surprise by setting \(\eta_t \to 1\) (possibly due to the change of context), or (2) fully incorporate the last surprise by setting \(\eta_t \to 1\) (possibly as the token is highly relevant to its recent past tokens).

**Objective.** Our above surprise metric is based on a loss function \(\mathcal{L}(.;.)\), which is the objective that our memory is learning to act as at test time. That is, our memory module is a meta model that learns a function based on the loss function \(\mathcal{L}(.;.)\).
# 3 Learning to Memorize at Test Time

To overcome the lack of long-term memory and to enable the model to learn, forget, and retrieve information, in this section, we present a neural long-term memory module, which is a meta model that learns to memorize at test time. In Section 3.1, we first discuss the motivation and the design of the neural memory. In Section 3.2, we discuss how our architecture design can benefit from a fast and parallelizable training. Finally, in Section 3.3, we augment our architecture using persistent memory module, in which we use learnable but data-independent parameters to learn meta information about the task.

## 3.1 Long-term Memory

To design a neural long-term memory module, we need a model that can encode the abstraction of the past history into its parameters. An example of this can be LLMs that are shown to be memorizing their training data (Leybzon and Kervadec 2024; Schwarzschild et al. 2024; Staab et al. 2024). Therefore, a simple idea is to train a neural network and expect it to memorize its training data. Memorization, however, has almost always been known as an undesirable phenomena in neural networks as it limits the model generalization (Bayat et al. 2024), and it causes privacy concerns (Staab et al. 2024), and so results in poor performance at test time. Moreover, the memorization of the training data might not be helpful at test time, in which the data might be out-of-distribution. We argue that, instead, we need an online meta-model that learns how to memorize/forget the data at test time. In this setup, the model is learning a function that is capable of memorization, but it is not overwriting to the training data, resulting in a better generalization at test time.

**Learning Process and Surprise Metric.** The key idea to train a long-term memory is to treat its training as an online learning problem, in which we aim to compress the past information \(x_1, \ldots, x_{t-1}\) into the parameters of our long-term neural memory module \(M_t\). As discussed earlier, we argue that violates the expectations (i.e., is surprising) is more memorable for humans (Mandler 2014). Inspired by this, a simple definition of surprise for a model can be its gradient with respect to the memory. The larger the gradient is, the more different the input data is from the past data. Accordingly, using this surprise score, we can update the memory as:

\[
M_t = M_{t-1} - \theta_t \nabla \mathcal{L}(M_{t-1:x_t})
\]

This surprise metric, however, can result in missing important information that comes after a big surprising moment. That is, the gradient can become extremely small after several surprising steps, leading to stocking in a flat area (i.e., local minima), and missing information about some parts of the sequence. From the human memory perspective, an event might not consistently surprise us through a long period of time although it is memorable. The reason is that the initial moment is surprising enough to get our attention through a long time frame, leading to memorizing the entire time frame. To improve the above surprise metric (Equation 8), we break the surprise metric into (1) past surprise, which measures the surprise amount of a very recent past; and (2) momentary surprise, which measures the surprise of incoming data:

\[
M_t = M_{t-1} + S_t
\]

\[
S_t = \eta_t S_{t-1} - \theta_t \nabla \mathcal{L} (M_{t-1:x_t}).
\]

Interestingly, this formulation is similar to gradient descent with momentum, where \(S_t\) is the momentum element. Therefore, the momentum here acts as a memory of surprise across time (sequence length). In this formulation, the term \(\eta_t\) is a data-dependent surprise decay (a function of \(x_t\)), controlling how our surprise decays over time, and the term \(\theta_t\) is controlling how much of momentary surprise should be incorporated into the final surprise metric in a data-dependent manner. This data-dependency is particularly important in this design: While surprise of previous tokens might be needed to affect the surprise of the next token, it is mostly valid if all tokens are relevant and are in the same context. Accordingly, a data-dependent \(n\) can control if memory needs to: (1) ignore the last surprise by setting \(\eta_t \to 1\) (possibly due to the change of context), or (2) fully incorporate the last surprise by setting \(\eta_t \to 1\) (possibly as the token is highly relevant to its recent past tokens).

**Objective.** Our above surprise metric is based on a loss function \(\mathcal{L}(.;.)\), which is the objective that our memory is learning to act as at test time. That is, our memory module is a meta model that learns a function based on the loss function \(\mathcal{L}(.;.)\).
In this work, we focus on associative memory, in which we aim to store the past data as the pairs of keys and values. Given \( x_t \), similar to Transformers (Vaswani et al. 2017), we use two linear layers to project \( x_t \) into a key and value:

\[
k_t = x_tW_k,
\]
\[
v_t = x_tW_v,
\]

where \( W_k \) and \( W_v \in \mathbb{R}^{d_m \times d_a} \). Next, we expect our memory module to learn the associations between keys and values. To this end, we define the loss as follows:

\[
\ell(M_{t-1};x_t) = \|M_{t-1} - v_t\|^2_2
\]

By optimizing the above loss function in the inner-loop of our meta model (memory), the model learns how to memorize the mapping between keys and values at test time. Note that, similar to meta-learning models (Nichol 2018; Zintgraf et al. 2019), training of the memory is in the inner-loop, and so parameters \( W_k \) and \( W_v \) are hyperparameters in the above loss function. Accordingly, in the inner loop, we optimize \( M \)’s weights, while in the outer-loop, we optimize other parameters of the entire architecture.

**Forgetting Mechanism.** When dealing with very large sequences (e.g., millions of tokens), it is crucial to manage which past information should be forgotten—even with a deep or a very large matrix-valued memory. To this end, we use an adaptive forgetting mechanism that allows the memory to forget the information that is not needed anymore, resulting in better managing the memory’s limited capacity. That is, given the next token \( x_t \), we modify the update rule as:

\[
M_t = (1 - \alpha_t)M_{t-1} + S_t,
\]

\[
S_t = \eta_t S_{t-1} - \theta_t \nabla \ell(M_{t-1};x_t),
\]

where \( \alpha_t \in [0, 1] \) is the gating mechanism that flexibly controls the memory; i.e., decides how much information should be forgotten. For example, it can update the memory without affecting the past abstraction by letting \( \alpha_t \to 0 \), and can clear the entire memory by letting \( \alpha_t \to 1 \). Later in this section, we show that this weight decay mechanism is closely related to the gating mechanism in modern RNNs (Dao and Gu 2024; Orvieto et al. 2023).

**Memory Architecture.** In this paper, we focus on simple MLPs with \( L_M \geq 1 \) as the architecture of our long-term memory. The main reason behind this choice is that we want to focus on better motivating the design of the long-term memory and ways that it can be incorporated into an architecture. However, our formulation and architectural design opens a new research direction to design neural architectures that are more effective and efficient in memorization of data. Recently, there has been a promising line of work to design such architectures (Berges et al. 2024; Cetin et al. 2024; Zhang et al. 2024), which incorporating them into our framework (i.e., replacing simple MLPs with such architectures) can be interesting future work.

When using vector-valued or matrix-valued memory (De et al. 2024; Orvieto et al. 2023; S. Yang, B. Wang, Shen, et al. 2024), the memory module is compressing the past data and fit it into a line. That is, from the meta learning or online learning perspective (Yuan Sun et al. 2024), using a matrix-valued memory \( M = W \in \mathbb{R}^{d_e \times d_a} \) is equivalent to optimize \( \|W_{t-1};x_t - W_{t-1}v_t\|^2_2 \), which is an online linear regression objective and so the optimal solution assumes the underlying dependency of historical data is linear. On the other hand, we argue that deep memory modules (i.e., \( L_M \geq 2 \)). Aligning with the theoretical results that MLPs with at least two layers are strictly more expressive than linear models (Hornik, Stinchcombe, and White 1989), in Section 5.5, we show that deep memory modules are more effective in practice.

**Retrieving a Memory.** In the above, we discuss how one can design and train a long-term memory module that learns to memorize at test time. A key remaining question is: How can we retrieve information from the memory? We simply use the forward pass without weight update (i.e., inference) to retrieve a memory to correspond to a query. Formally, given an input \( x_t \), we use a linear layer \( W_q \) to project the input, i.e., \( q_t = x_tW_q \) and retrieve the corresponding (or useful) information from the memory \( y_t \) by:

\[
y_t = M^*(q_t).
\]
![Figure 1: The illustration of how the training of neural memory can be done in parallel and using matmuls.](attachment://image1.png)

### 3.2 How to Parallelize the Long-term Memory Training

As discussed above, the design of our long-term memory module is equivalent to training a meta model by optimizing the associative memory loss function \(M_{(-i:X)} = ||M_{-1} - k_y||^2_2\) using gradient descent with momentum and weight decay. Therefore, in theory, the training of long-term memory module requires \(O(N)\) FLOPs, where \(N\) is the sequence length. However, in practice, we need to parallelize the training process and to fully take advantage of hardware accelerators (e.g., TPUs, GPUs), we need to tensorize the process and use more matmuls.

Next, we show that calculating the weights in the inner loop with mini-batch gradient descent, data-dependent learning rate, and weight decay can be reformulated so that it uses only matmuls and sum. We build upon the work of Yu Sun et al. (2024) that shows forward pass of a model optimizing with the mini-batch gradient descent (with constant learning rate) can be calculated using matmuls. We can split the sequence into chunks of size \(b \geq 1\), and write the mini-batch gradient descent as:

\[
M_t = (1 - \alpha_t)M_{t-1} - \theta \nabla_{(M_{-1:X})} = \beta_t M_0 - \sum_{i=1}^{b} \frac{\beta_t}{\beta_i} \nabla_t (M_{r}; x_t) \tag{16}
\]

where \(t' = t \mod(t, b)\), and \(\beta_i = \prod_{j=1}^{i-1}(1 - \alpha_j)\). For the sake of simplicity, we focus on the first chunk, i.e., \(t = b\) and so \(t' = 0\). Also, we explain the process for the case that \(M_t = W_i\) is linear. The process for MLPs with \(Np = 2\) is similar. Using our loss function, we have:

\[
\nabla_{(W_0; x_t)} = (W_{0:X} - x_t)X^T = \sum_{i=1}^{b} \frac{\beta_t}{\beta_i} \nabla_{(W_0; x_t)} = \Theta_b (W - X)^T, \tag{17}
\]

where \(\Theta_b = \text{diag}(\left| \Theta_0 \cdots \Theta_b \right|)\) and \(B_b\) is defined analogously on \(\beta_b\). Note that, we do not need to store all \( \Theta_b\) and \(B_k\) for \(k = 1, \ldots, N\); instead, we store these matrices for each chunk, resulting in less memory. Next, we extend this representation so we can also incorporate the momentum term. In a chunk wise gradient descent with momentum, if we look at the momentum term, we have:

\[
S_t = \eta_t S_{t-1} - \theta u_t, \tag{18}
\]

where \(u_t = \nabla_{(M_t; x_t)}\). Note that, we can compute all \(u_t\) at the same time, and so Equation 18 is a linear recurrence with \(u_t\) as an input, \(S_t\) as the hidden state, and \(\eta_t\) as input-dependent transition value. Accordingly, we can use parallel associative scan (J. T. Smith, Warrington, and Linderman 2023) to calculate \(S_t\) in this chunk.

Parameters as the Function of Chunks. Instead of making parameters like \(\alpha_t\), \(\theta_t\), and \(\eta_t\) input-dependent (i.e., a function of token \(x_t\)), we can make them functions of their chunk. Despite losing expressive power, this formulation can help to make the training even faster. In this case, we are using the same value for each of \(\alpha\), \(\theta\), and \(\eta\) in each chunk. Accordingly, in Equation 17, we are able to use a single equation. Similarly we can make Equation 18 faster. That is, when \(\eta\) and \(\theta\) are learnable but time-invariant inside each chunk, this equation becomes a linear time-invariant system (LTI), which can be computed by a global convolution (Gu, Goel, and Re 2022). In our experiments, we make these parameters as the functions of tokens. However, such simplifications (i.e., as the function of chunks) can be the interest of future work to training larger models in more efficient manner.
![Figure 2: Memory as a Context (MAC) Architecture. This architecture includes three branches of (1) core, (2) contextual (long-term) memory, and (3) persistent memory. The core branch concatenates the corresponding long-term and persistent memories with the input sequence. Next, attention performs on the sequence and decides what part of the information should store in the long-term memory. At the test time, parameters correspond to contextual memory are still learning, parameters correspond to the core branch are responsible for in-context learning, and parameters of persistent memory are responsible to store the knowledge about tasks and are so fixed.](https://via.placeholder.com/150)

### 3.3 Persistent Memory

Our long-term memory can also be seen as a contextual memory, meaning that the output is fully dependent on the context. Therefore, in addition to our long-term memory, we also use a set of learnable but input-independent parameters to act as task-related memory. This type of memory has been referred to as persistent or meta-memory in the literature (X. Dong et al. 2024; Sukhabaatar, Grave, et al. 2019). Given \( N_p \geq 1 \), we use learnable parameters \( P = [p_1, p_2, \ldots, p_{N_p}] \) and append it to the start of our sequence: i.e., given a context window size of \( N \), we modify the input as:

\[
x_{\text{new}} = [p_1, p_2, \ldots, p_{N_p}] \Vert x,
\]

where \( \Vert \) is concatenation. Next, we discuss the motivation of persistent memory from three perspectives:

**Memory Perspective.** As discussed earlier, our neural long-term memory is a contextual memory, in which all parameters are input-dependent. An effective memory system, however, also needs input-independent parameters to store the abstraction of the task knowledge. That is, mastering a task requires the memorization of the knowledge that how the task can be done, and these parameters are responsible for storing such knowledge.

**Feedforward Network Perspective.** In the Transformer architectures, there are fully connected layers after the attention module, which are shown to be similar to attention weights but with data-independent parameters. That is, Sukhabaatar, Grave, et al. (2019) showed that replacing the ReLU in fully connected layers with Softmax can result in an attention-like weights, in which weights are data-independent:

\[
\text{FFN}(x) = W_f \text{Softmax}(W_k x).
\]

In fact, \( W_k \) and \( W_y \) are acting similar to \( K \) and \( V \) matrices in the attention module when they are input-independent. The persistent memory weights are expected to have the same functionality, meaning that using them in the first part of the sequence leads to having input-independent attention weights (Sukhabaatar, Grave, et al. 2019).

**Technical Perspective.** Attention with causal masks has implicit bias toward initial tokens in the sequence, and so attention weights are almost always highly active for initial tokens, resulting in performance damage. From the technical perspective, these learnable parameters at the start of the sequence can mitigate such effect by redistributing the attention weights more effectively (Han et al. 2024; Xiao et al. 2024).
![Figure 2: Memory as a Context (MAC) Architecture. This architecture includes three branches of (1) core, (2) contextual (long-term) memory, and (3) persistent memory. The core branch concatenates the corresponding long-term and persistent memories with the input sequence. Next, attention performs on the sequence and decides what part of the information should store in the long-term memory. At the test time, parameters correspond to contextual memory are still learning, parameters correspond to the core branch are responsible for in-context learning, and parameters of persistent memory are responsible to store the knowledge about tasks and are so fixed.](https://via.placeholder.com/150)

### 3.3 Persistent Memory

Our long-term memory can also be seen as a contextual memory, meaning that the output is fully dependent on the context. Therefore, in addition to our long-term memory, we also use a set of learnable but input-independent parameters to act as task-related memory. This type of memory has been referred to as persistent or meta-memory in the literature (X. Dong et al. 2024; Sukhabaatar, Grave, et al. 2019). Given \( N_p \geq 1 \), we use learnable parameters \( P = [p_1, p_2, \ldots, p_{N_p}] \) and append it to the start of our sequence: i.e., given a context window size of \( N \), we modify the input as:

\[
x_{\text{new}} = [p_1, p_2, \ldots, p_{N_p}] \Vert x,
\]

where \( \Vert \) is concatenation. Next, we discuss the motivation of persistent memory from three perspectives:

**Memory Perspective.** As discussed earlier, our neural long-term memory is a contextual memory, in which all parameters are input-dependent. An effective memory system, however, also needs input-independent parameters to store the abstraction of the task knowledge. That is, mastering a task requires the memorization of the knowledge that how the task can be done, and these parameters are responsible for storing such knowledge.

**Feedforward Network Perspective.** In the Transformer architectures, there are fully connected layers after the attention module, which are shown to be similar to attention weights but with data-independent parameters. That is, Sukhabaatar, Grave, et al. (2019) showed that replacing the ReLU in fully connected layers with Softmax can result in an attention-like weights, in which weights are data-independent:

\[
\text{FFN}(x) = W_f \text{Softmax}(W_k x).
\]

In fact, \( W_k \) and \( W_y \) are acting similar to \( K \) and \( V \) matrices in the attention module when they are input-independent. The persistent memory weights are expected to have the same functionality, meaning that using them in the first part of the sequence leads to having input-independent attention weights (Sukhabaatar, Grave, et al. 2019).

**Technical Perspective.** Attention with causal masks has implicit bias toward initial tokens in the sequence, and so attention weights are almost always highly active for initial tokens, resulting in performance damage. From the technical perspective, these learnable parameters at the start of the sequence can mitigate such effect by redistributing the attention weights more effectively (Han et al. 2024; Xiao et al. 2024).
# 4 How to Incorporate Memory?

An important question that remained unanswered is: How one can effectively and efficiently incorporate the designed neural memory into a deep learning architecture? As discussed earlier, from a memory perspective, the pair of K and V matrices in transformers can be interpreted as an associative memory block. Due to their accommodating of dependencies and so their limited context window, we interpret them as short-term memory modules, attending to the current context window size. On the other hand, our neural memory with the ability to continuously learn from data and store it in its weights can play the role of a long-term memory. In this section, we aim to answer the above question by proposing three different variants of Titans. Later in our experiments, we show that each of these variants has its own advantages/disadvantages and also can show a trade-off between the efficiency and effectiveness in very long-contexts.

## 4.1 Memory as a Context

In the first architecture design (see Figure 2), we treat the memory as a context to the current information. That is, given a long sequence \( x \in \mathbb{R}^{N \times d_{emb}} \), we first chunk the sequence into fixed-size segments \( s^{(l)} \) for \( i = 1, \ldots, N/C \). Given the incoming segment \( s^{(l)} \), we consider it as the current context and its past segment as the historical information. Therefore, let \( M_{t-1} \) be the state of long-term memory before segment \( s^{(l)} \), we use the input context as the query to the memory \( M_{t-1} \) to retrieve the corresponding information from the long-term memory. That is, we retrieve the past information that corresponds to \( s^{(l)} \) as:
\[
h_t = M_{t-1}(q_t),
\]
where \( q_t = s^{(l)}W_q \). Next, we use this historical information along with our persistent memory parameters as the input sequence to the attention module:
\[
\bar{s}^{(l)} = [p_1 \; p_2 \; \cdots \; p_N \; || \; h_t \; || \; s^{(l)}],
\]
\[
y_t = \text{Attn}\left( \bar{s}^{(l)} \right).
\]

The structure of the attention map over the entire sequence is shown in Figure 3a. We then use \( y_t \) to update the long-term memory module for the next segment and the final output:
\[
M_t = M_{t-1}(y_t),
\]
\[
o_t = y_t \odot M_t(y_t).
\]

Note that, in the above, we are updating the weight of \( M_{t-1} \) through forward pass.

This architecture has two key advantages: (1) Attention by having both historical and current context, has the ability to decides whether given the current data, the long-term memory information is needed. (2) The attention module helps.
![Memory as a Gate (MAG) Architecture](https://example.com/memory_architecture.png)

Figure 4: **Memory as a Gate (MAG) Architecture.** This architecture, similarly, has the three branches of (1) core, (2) contextual memory, and (3) persistent memory. It, however, incorporates only persistent memory into the context and combine memory with the core branch using a gating mechanism. At test time, the behavior is the same as Figure 2.

the long-term memory to store only useful information from the current context. That is, not all tokens in each segment are useful and memorizing all of them can result in memory overflow. Therefore, attention is helping the memory to understand which information is useful, better managing the memory capacity. (3) At test time: (i) persistent memory parameters are fixed as they encode the knowledge about the task, which should not be changed; (ii) the attention module weights are in-context learner; and (iii) the long-term memory module is still learning (memorizing) the information at test time. That is, we update the weights of the neural memory even at test time as weights are encoding the abstraction of long past.

## 4.2 Gated Memory

In the next variant (see Figure 4), in one branch, we directly use the input data to update the long-term memory, and in the second branch, we use a sliding window attention (SWA):

\[
\hat{x} = [p_1, p_2, \ldots, p_N] \| x,
\]

\[
y = SW\text{-}Attn^*(\hat{x}),
\]

\[
o = g(M(\hat{x})),
\]

where \(SW\text{-}Attn^*\) is sliding window attention with prefix (see Figure 3b). Note that, contrary to the previous design, we are not segmenting the input data. Also, we abuse the notation and use \(M(x)\) to refer to the final output of the memory after all recursion over the tokens of the sequence. In the above equation, \(o\) can be any non-linear gating. In our experiments, we normalize the outputs \(y\) and \(M(\hat{x})\) using learnable vector-valued weights, followed by a non-linearity \(\sigma(\cdot)\).

The overall attention of this design is shown in Figure 3b. In this design, sliding window attention is acting as a precise short-term memory, while the neural memory module is acting as a fading memory for the model. This architecture design can also be seen as a multi-head architecture where the structure of heads are different (X. Dong et al. 2024).

## 4.3 Memory as a Layer

The last variant uses the neural Memory As a Layer (MAL) of a deep neural network (see Figure 5). This architecture design is more common in the literature, where the hybrid models stack recurrent models with full or sliding window attentions. Given input \(x\), we have:

\[
\hat{x} = [p_1, p_2, \ldots, p_N] \| x,
\]

\[
y = M(\hat{x}),
\]

\[
o = SW\text{-}Attn(y),
\]
# 5 Experiments

Next, we evaluate the performance of Titans and its variants in language modeling, commonsense reasoning, needle in haystack, DNA modeling, and time series forecasting tasks1. In more details, in this section, we answer the following empirical questions: (1) How do Titans perform compared to baselines in downstream tasks? (see §5.2,
...

Figure 5: Memory as a Layer (MAL) Architecture. In this architecture, the memory layer is responsible to compress the past and current context before the attention module.

where SW-Attn is sliding window attention. The main drawback of this design is that the power of the model is limited by each of the layers and so it cannot take advantage of the complementary data processing of attention and neural memory module. In our experiments, for evaluating memory in this design, we use a similar architecture as H3 (D. Y. Fu et al. 2023), where we replace the sequence model with our neural memory module (LMM).

Memory Without Attention. Although in the above, we discussed MAL as the combination of LMMs and attention in a sequential manner, one simple variant of MAL is to treat LMM as a sequence model without any attention. From the memory perspective, as discussed in Section 1, we expect each part of the memory system to work independently, even if other components are disturbed. Therefore, a long-term memory module should still be a powerful model even without short-term memory (i.e., attention). We refer to this variant as LMM or Titans (LMM) in our experiments. We provide additional discussions on the connection of Titans and other modern recurrent models in Appendix C.

## 4.4 Architectural Details

For the sake of simplicity and presentation, we avoid discussing the implementation details like using residual connection, gating with linear layer, and normalization. In all blocks, we use residual connections. In our implementation, we use SiLU() activation (Elfwing, Uchibe, and Doya 2018) as the non-linear activation for computing query, key, and values and normalize queries and keys using l2-norm.

### Convolution

Following the recent modern linear recurrent models (Gu and Dao 2024; S. Yang, Kautz, and HataMIDZEH 2024), we incorporate a 1D depthwise-separable convolution layer after each of the query, key, and value projections. While not significantly affecting the performance, these 1D convolutions have shown performance improvement and are also computationally efficient.

### Gating

We also follow the recent architectures that use normalization and gating with a linear layer before the final output projection (Mehta et al. 2023).

#### Theorem 4.1

Contrary to Transformers, diagonal linear recurrent models, and DeltaNet, all of which are limited to T^C (Merrill, Petty, and Sabharwal 2024), Titans are capable of solving problems beyond T^C, meaning that Titans are theoretically more expressive than Transformers and most modern linear recurrent models in state tracking tasks.
# 5.1 Experimental Setup

Models. In our experiments, we focus on the three variants of Titans, which we refer to as: Titans with (1) Memory as a Context (MAC), (2) Memory as a Gate (MAG), and (3) Memory as a Layer (MAL) as well as (4) neural memory module alone. The reason behind using our long-term memory as a separate module is based on our definition of learning. As discussed in Section 1, we define learning a process for acquiring effective and useful memory. Accordingly, we expect our long-term memory to effectively learn from data, even without attention. For each of these models, we consider four scales with: (i) 170M, (ii) 340M, (iii) 400M, and (iv) 760M parameters. While the first three are trained on 15B tokens sampled from FineWeb-Edu dataset (Penedo et al. 2024), the last one is trained on 30B tokens from the same dataset.

Baselines. We compare our models with the state-of-the-art linear recurrent models, Transformers, and hybrid models (recurrent + attention). More specifically in language tasks, we compare with Transformer+† (Touvron et al. 2023), RetNet Yutao Sun et al. 2023), Gated Linear Attention (GLA) (S. Yang, B. Wang, Shen, et al. 2024), Mamba (Gu and Dao 2024), Mambo (Dao and Gu 2024), DeltaNet (S. Yang, Kautz, and Hatamizadeh 2024), In needle in haystack tasks, we also compare with GPT-4 (Achiam et al. 2023), Llama3 with RAG (Touvron et al. 2023), RecurrentGemma-9B (Botev et al. 2024), and Mistral (Jiang et al. 2023) models, all of which are provided in the benchmark (Yuri Kuraev et al. 2024). In time series tasks, we compare with Mamba-based (Behrouz, Santacatterina, and Zabih 2024), Transformer-based (Y. Liu et al. 2023; Nie et al. 2022; Yunhao Zhang and Yan 2023), and linear models (Das et al. 2023; Z. Li et al. 2023; H. Wu et al. 2023; Zeng et al. 2023).

Training. In the training, we follow the procedure of S. Yang, Kautz, and Hatamizadeh (2024), and use Llama 2 models as a preference of a key size of 32K and use training length of tokens. We employ Adam optimizer with learning rate of 4e-4 with cosine annealing schedule with batch size of 0.5M tokens, and weight decay of 0.1.

# 5.2 Language Modeling

We first focus on the perplexity in language modeling and also commonsense reasoning tasks. The results for Titans' variants and also baselines with three different sizes of 340M, 400M, and 760M are reported in Table 1. Among non-hybrid models, including Transformer+†, our neural memory module achieves the best performance in both perplexity and accuracy measures. Comparing our neural memory module and TTT, which is also a gradient-based recurrent model can show us the importance of our weight decay as well as the momentum. As discussed earlier, the weight decay can be interpreted as a gating mechanism to forget the past data, when it is needed. Also, momentum can help us better manage the memory by providing additional memory for the surprise metric. While some baselines also take advantage of gating mechanism, e.g., Mamba, Mamba2, and Gated DeltaNet, the superior performance of our neural memory module shows the importance of both our surprise mechanism and having deep and non-linear memory. We further discuss the later in Section 5.5.

Comparing the hybrid models, we found that all three variants of Titans (MAC, MAG, and MAL) outperform both Samba (Mamba + attention) and Gated DeltaNet-H2 (Gated DeltaNet + attention). We attribute the superior performance of Titans (MAL) to the power of neural memory module as the architecture design and used attention all the same. Comparing Titans (MAG) and (MAC), we find that while their performance are close, MAC performs better when dealing with longer dependencies in the data. Interestingly, both MAG and MAC outperform MAL variant, which due to using the same models, we attribute this to the architecture design of these models. This finding is particularly important as the current hybrid models (except Hymba (X. Dong et al. 2024)) in the literature are using MAL-style combination of recurrent models and attention.

# 5.3 Needle in a Haystack

Scaling a model to longer context window is not always equivalent to being effective for very long sequences (Hsieh et al. 2024). The needle-in-a-haystack (NIAH) task is designed to measure the actual effective context length of models. In this task, we evaluate the model on retrieving a piece of information (i.e., the "needle") from long distractor texts (i.e.,
# 5.1 Experimental Setup

Models. In our experiments, we focus on the three variants of Titans, which we refer to as: Titans with (1) Memory as a Context (MAC), (2) Memory as a Gate (MAG), and (3) Memory as a Layer (MAL) as well as (4) neural memory module alone. The reason behind using our long-term memory as a separate module is based on our definition of learning. As discussed in Section 1, we define learning a process for acquiring effective and useful memory. Accordingly, we expect our long-term memory to effectively learn from data, even without attention. For each of these models, we consider four scales with: (i) 170M, (ii) 340M, (iii) 400M, and (iv) 760M parameters. While the first three are trained on 15B tokens sampled from FineWeb-Edu dataset (Penedo et al. 2024), the last one is trained on 30B tokens from the same dataset.

Baselines. We compare our models with the state-of-the-art linear recurrent models, Transformers, and hybrid models (recurrent + attention). More specifically in language tasks, we compare with Transformer+† (Touvron et al. 2023), RetNet Yutao Sun et al. 2023), Gated Linear Attention (GLA) (S. Yang, B. Wang, Shen, et al. 2024), Mamba (Gu and Dao 2024), Mambo (Dao and Gu 2024), DeltaNet (S. Yang, Kautz, and Hatamizadeh 2024), In needle in haystack tasks, we also compare with GPT-4 (Achiam et al. 2023), Llama3 with RAG (Touvron et al. 2023), RecurrentGemma-9B (Botev et al. 2024), and Mistral (Jiang et al. 2023) models, all of which are provided in the benchmark (Yuri Kuraev et al. 2024). In time series tasks, we compare with Mamba-based (Behrouz, Santacatterina, and Zabih 2024), Transformer-based (Y. Liu et al. 2023; Nie et al. 2022; Yunhao Zhang and Yan 2023), and linear models (Das et al. 2023; Z. Li et al. 2023; H. Wu et al. 2023; Zeng et al. 2023).

Training. In the training, we follow the procedure of S. Yang, Kautz, and Hatamizadeh (2024), and use Llama 2 models as a preference of a key size of 32K and use training length of tokens. We employ Adam optimizer with learning rate of 4e-4 with cosine annealing schedule with batch size of 0.5M tokens, and weight decay of 0.1.

# 5.2 Language Modeling

We first focus on the perplexity in language modeling and also commonsense reasoning tasks. The results for Titans' variants and also baselines with three different sizes of 340M, 400M, and 760M are reported in Table 1. Among non-hybrid models, including Transformer+†, our neural memory module achieves the best performance in both perplexity and accuracy measures. Comparing our neural memory module and TTT, which is also a gradient-based recurrent model can show us the importance of our weight decay as well as the momentum. As discussed earlier, the weight decay can be interpreted as a gating mechanism to forget the past data, when it is needed. Also, momentum can help us better manage the memory by providing additional memory for the surprise metric. While some baselines also take advantage of gating mechanism, e.g., Mamba, Mamba2, and Gated DeltaNet, the superior performance of our neural memory module shows the importance of both our surprise mechanism and having deep and non-linear memory. We further discuss the later in Section 5.5.

Comparing the hybrid models, we found that all three variants of Titans (MAC, MAG, and MAL) outperform both Samba (Mamba + attention) and Gated DeltaNet-H2 (Gated DeltaNet + attention). We attribute the superior performance of Titans (MAL) to the power of neural memory module as the architecture design and used attention all the same. Comparing Titans (MAG) and (MAC), we find that while their performance are close, MAC performs better when dealing with longer dependencies in the data. Interestingly, both MAG and MAC outperform MAL variant, which due to using the same models, we attribute this to the architecture design of these models. This finding is particularly important as the current hybrid models (except Hymba (X. Dong et al. 2024)) in the literature are using MAL-style combination of recurrent models and attention.

# 5.3 Needle in a Haystack

Scaling a model to longer context window is not always equivalent to being effective for very long sequences (Hsieh et al. 2024). The needle-in-a-haystack (NIAH) task is designed to measure the actual effective context length of models. In this task, we evaluate the model on retrieving a piece of information (i.e., the "needle") from long distractor texts (i.e.,
| Model | Wiki pp | LMB. acc ↑ | LMB. acc ↑ | PIQA acc ↑ | Hella. Wino. ARC-e acc ↑ | ARC-c acc ↑ | SIQA acc ↑ | BoolQ acc ↑ | Avg. ↑ |
|----------------------|---------|-------------|-------------|-------------|---------------------------|--------------|-------------|--------------|---------|
| Transformer++ | 31.52 | 40.70 | 62.98 | 34.76 | 50.35 | 45.21 | 24.56 | 58.84 | 42.92 |
| RetNet | 32.50 | 49.78 | 28.74 | 62.61 | 34.15 | 50.91 | 44.27 | 23.26 | 59.72 |
| GLA | 28.51 | 43.02 | 28.73 | 64.05 | 39.56 | 50.00 | 54.19 | 24.97 | 53.49 |
| Mamba | 30.80 | 40.21 | 29.94 | 63.79 | 35.88 | 49.24 | 26.46 | 35.41 | 60.07 |
| DeltaNet | 26.78 | 47.30 | 28.43 | 63.52 | 55.99 | 45.23 | 25.37 | 36.79 | 56.79 |
| TTT | 27.01 | 34.19 | 30.63 | 39.71 | 50.08 | 53.01 | 26.17 | 32.39 | 59.83 |
| Gated DeltaNet | 27.10 | 39.94 | 34.11 | 63.08 | 51.60 | 55.28 | 26.77 | 39.54 | 45.42 |
| Titans (LM) | 26.18 | 29.97 | 34.75 | 39.61 | 51.85 | 55.60 | 28.14 | 52.99 | 46.17 |
| Titans (MAC) | 25.43 | 28.11 | 36.00 | 63.52 | 51.21 | 58.17 | 29.00 | 38.43 | 47.36 |
| Titans (MAG) | 25.07 | 28.72 | 36.71 | 46.50 | 52.72 | 27.12 | 19.78 | 16.30 | 50.61 |
| Titans (MAL) | 24.69 | 28.50 | 35.74 | 49.41 | 51.67 | 26.58 | 28.21 | 38.14 | 57.32 |

| Model | Wiki pp | LMB. acc ↑ | LMB. acc ↑ | PIQA acc ↑ | Hella. Wino. ARC-e acc ↑ | ARC-c acc ↑ | SIQA acc ↑ | BoolQ acc ↑ | Avg. ↑ |
|----------------------|---------|-------------|-------------|-------------|---------------------------|--------------|-------------|--------------|---------|
| Transformer++ | 30.63 | 37.37 | 29.64 | 64.27 | 37.72 | 51.53 | 54.95 | 27.36 | 38.07 |
| RetNet | 29.92 | 46.83 | 26.94 | 36.97 | 51.88 | 56.01 | 27.55 | 37.30 | 59.62 |
| HGRN2 | 32.32 | 47.14 | 26.12 | 54.52 | 35.57 | 25.97 | 25.71 | 37.58 | 45.09 |
| GLA | 27.92 | 36.86 | 27.86 | 53.71 | 49.66 | 26.36 | 28.36 | 19.84 | 54.54 |
| Mamba | 26.34 | 33.19 | 62.57 | 37.93 | 52.48 | 29.70 | 37.92 | 60.29 | 46.91 |
| DeltaNet | 26.97 | 41.09 | 41.80 | 40.99 | 51.96 | 54.33 | 30.33 | 23.59 | 59.36 |
| Gated DeltaNet-H2 | 26.11 | 31.52 | 39.16 | 47.37 | 43.59 | 52.30 | 29.09 | 37.02 | 54.37 |
| Titans (LM) | 25.03 | 28.99 | 25.80 | 49.53 | 48.62 | 52.27 | 63.41 | 35.84 | 47.83 |
| Titans (MAC) | 25.61 | 27.73 | 29.66 | 41.18 | 52.00 | 29.09 | 40.21 | 29.00 | 47.81 |
| Titans (MAG) | 25.39 | 27.91 | 28.76 | 40.42 | 52.31 | 55.12 | 19.80 | 29.09 | 46.81 |
| Titans (MAL) | 23.97 | 27.89 | 36.84 | 40.74 | 52.26 | 29.71 | 38.92 | 58.40 | 47.87 |
| Model | S-NIAH-PK | | | | | | | | | | | | | S-NIAH-N | | | | | | | | | | | | | | S-NIAH-W | | | |
|---------------|-----------|------|------|------|------|------|------|------|------|------|------|------|------|-----------|------|------|------|------|------|------|------|------|------|------|------|------|------|-----------|------|------|------|
| | 2K | 4K | 8K | 16K | 2K | 4K | 8K | 16K | 2K | 4K | 8K | 16K | | 2K | 4K | 8K | 16K | 2K | 4K | 8K | 16K | 2K | 4K | 8K | 16K | | 2K | 4K | 8K | 16K |
|---------------|-----------|------|------|------|------|------|------|------|------|------|------|------|------|-----------|------|------|------|------|------|------|------|------|------|------|------|------|------|-----------|------|------|------|
| TTT | 98.4 | 98.8 | 98.0 | 88.4 | 60.2 | 36.6 | 10.2 | 4.4 | 78.8 | 28.0 | 4.4 | 0.0 | | 98.4 | 98.8 | 98.4 | 95.2 | 42.2 | 4.2 | 0.0 | | | | | | | | | | |
| Mamba2 | 96.6 | 61.4 | 31.0 | 5.4 | 84.5 | 55.8 | 14.2 | 0.0 | 42.2 | 4.2 | 0.0 | 0.0 | | 96.5 | 98.6 | 96.6 | 71.4 | 17.2 | 15.4 | 2.0 | | | | | | | | | | |
| DeltaNet | 96.8 | 98.5 | 96.6 | 71.4 | 47.2 | 15.5 | 14.8 | 5.4 | 82.0 | 6.0 | 2.0 | 1.6 | | 96.8 | 98.8 | 96.8 | 94.0 | 80.2 | 90.0 | 94.5 | | | | | | | | | | |
| Titans (LMM) | 99.8 | 98.4 | 98.2 | 96.2 | 100.0| 99.3 | 83.4 | 80.2 | 99.4 | 89.4 | 85.5 | 80.6 | | 99.8 | 98.4 | 97.6 | 97.2 | 97.8 | 98.2 | 95.6 | | | | | | | | | | |
| Titans (MAC) | 99.2 | 98.5 | 90.9 | 96.9 | 98.2 | 96.4 | 97.6 | 97.2 | 98.2 | 96.8 | 96.0 | 90.2 | | 99.4 | 97.4 | 97.4 | 97.2 | 98.6 | 96.0 | 92.0 | | | | | | | | | | |
| Titans (MAG) | 99.4 | 97.4 | 97.4 | 97.2 | 98.2 | 97.6 | 98.2 | 96.8 | 97.2 | 96.0 | 90.0 | 90.4 | | 99.8 | 98.6 | 98.8 | 96.4 | 98.0 | 97.4 | 92.0 | | | | | | | | | | |
| Titans (MAL) | 98.8 | 98.6 | 98.8 | 97.8 | 98.8 | 99.8 | 98.1 | 96.8 | 96.4 | 98.8 | 97.4 | 92.0 | | 99.4 | 98.6 | 98.8 | 97.4 | 96.4 | 98.8 | 90.4 | | | | | | | | | | |


### Figure 6: Performance of Titans and baselines on BABILong benchmark. Titans (MAC) outperforms all baselines, including extremely large models, e.g., GPT-4.

We can see a significant drop in performance when increasing the sequence length; (3) Compared to DeltaNet, although it is capable of removing memory using delta rule, it cannot erase the memory, lacking forgetting mechanism. Finally, as results are achieved with smaller models than using Titans variants, where the best results correspond to Titans.

### 5.4 BABILong Benchmark
In the previous section we discussed the results on a simple NIAH tasks where a single need needs to be retrieved. Although Titans showed better performance compared to baselines, their true advantage over very long sequences is still hidden. To this end, in this section, we use a harder task from BABILong benchmark (Yuri Kuratov et al., 2024), in which the model needs to reason across facts distributed in extremely long documents. Below we follow the original experimental setup and training process in the benchmark. There are two settings: (1) Few-shot setting, in which we use large pre-trained models, and (2) fine-tuning setting, where we fine-tune the MAC variant of Titans to compete with other fine-tuned baselines. The results for few-shot setting are reported in Figure 6. In this setup, we can see Titans outperform all baselines–i.e., Mamba2.8B (Gu and Dao 2024), RWKY-6-7B (Peng, Goldstein, et al. 2024), RecurrentGemma-9B (Botev et al. 2024), Gemma-9B (Team et al. 2024), Llama3.1-8B (Touvron et al. 2023), GPT-4, and GPT-4o-min (Achiam et al. 2023). These results are achieved while Titans (MAC) is having much less number of parameters than baselines.

In the fine-tuning setup, we compare the small fine-tuned version of Titans (MAC) with: (i) the fine-tuned version of small models (acting the same number of parameters as Titans) such as Mamba (Gu and Dao 2024), RWKY (Bulatev, Yury Kuratov, and Burstov 2022), (ii) large models with Retrieval-Augmented Generation (RAG) (P. Lewis et al. 2020) such as Llama3.1-8B (Touvron et al. 2023), and (iii) extremely large models such as GPT-4 (Achiam et al. 2023), Qwen2.5-7B (A. Yang et al. 2024), and Llama3.1-7B (Touvron et al. 2023). Baseline results are reported by (Yuri Kuratov et al. 2024). The performance results of Titans and baselines are reported in Figure 6. Titans outperform all models even extremely large models like GPT-4. Also, compared to Transformer-based memory models like RMT, Titans show better performance mainly due to their powerful memory. That is, RMT captures the historical data into 16 size vector-valued memory, while Titans with in-context online memory learner are capable of encoding the past into the parameters of the model. Interestingly, even...
### Figure 7: The effect of memory depth on the perplexity. Deeper long-term memory results in better scaling in longer sequences.

#### (a) 170M Parameters
![170M Parameters](#)

#### (b) 360M Parameters
![360M Parameters](#)

#### (c) 760M Parameters
![760M Parameters](#)

Table 3: Performance on long-term forecasting. The best results are highlighted.

| Neural Memory | Simba | iTransformer | Linear | PathTST | Crossformer | TIDE | TimeSNet | DLinear |
|---------------|-------|--------------|--------|---------|-------------|------|----------|---------|
| MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE |
| ETTh1 | 0.358 | 0.367 | 0.396 | 0.407 | 0.407 | 0.360| 0.419 | 0.406 |
| ETTh2 | 0.421 | 0.371 | 0.323 | 0.326 | 0.327 | 0.296| 0.291 | 0.266 |
| ETH1 | 0.366 | 0.362 | 0.354 | 0.367 | 0.372 | 0.352| 0.351 | 0.353 |
| ETH2 | 0.396 | 0.392 | 0.372 | 0.388 | 0.388 | 0.361| 0.320 | 0.285 |
| Traffic | 0.231 | 0.225 | 0.258 | 0.278 | 0.272 | 0.291| 0.287 | 0.305 |
| Weather | 0.231 | 0.225 | 0.258 | 0.276 | 0.272 | 0.291| 0.297 | 0.315 |

#### 5.5 The Effect of Deep Memory
In this section, we evaluate the effect of deep memory in both wall-clock training time and model performance^2. To this end, we focus on different variants of our neural memory module, where \( L_M = 1, 2, 3, 4 \). We also use Mamba as a baseline for the model performance. For a fair comparison, we use the same training process for all models and train them on a subset of the Pile dataset (L. Gao et al. 2020).

We report the perplexity of our models and baselines as the function of the sequence length in Figure 7. Interestingly, with the increase of memory depth, \( L_M \), the model can achieve better perplexity over all sequence length. Also, deeper memory modules are more robust to the sequence length when the model has less number of parameters. With the increase of the number of parameters, all models show better performance on longer sequences.

We also evaluate the effect of memory depth \( (L_M = 1, 2, 3, 4) \) on the training throughput. We report the training throughput (the number of tokens per second) as the function of sequence length in Figure 8. All models scale linearly with respect to the context length (i.e., constant trend in the number of tokens per second with respect to sequence length). Also, by increasing the memory depth, as expected, we can see a linear trend that a deeper memory results in a slower training. Therefore, it is not always efficient to use deeper memory modules, showing a trade-off between effectiveness and efficiency.

### Figure 8: The effect of memory depth on training throughput
![Memory depth effect](#)

#### 5.6 Time Series Forecasting
To show the effectiveness of our memory module in a broader tasks, we also evaluate its performance in time series forecasting tasks. To this end, we use the Simba framework (Patro and Agneeswaran 2024) for time series forecasting, and

### Figure 7: The effect of memory depth on the perplexity. Deeper long-term memory results in better scaling in longer sequences.

#### (a) 170M Parameters
![170M Parameters](#)

#### (b) 360M Parameters
![360M Parameters](#)

#### (c) 760M Parameters
![760M Parameters](#)

Table 3: Performance on long-term forecasting. The best results are highlighted.

| Neural Memory | Simba | iTransformer | Linear | PathTST | Crossformer | TIDE | TimeSNet | DLinear |
|---------------|-------|--------------|--------|---------|-------------|------|----------|---------|
| MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE |
| ETTh1 | 0.358 | 0.367 | 0.396 | 0.407 | 0.407 | 0.360| 0.419 | 0.406 |
| ETTh2 | 0.421 | 0.371 | 0.323 | 0.326 | 0.327 | 0.296| 0.291 | 0.266 |
| ETH1 | 0.366 | 0.362 | 0.354 | 0.367 | 0.372 | 0.352| 0.351 | 0.353 |
| ETH2 | 0.396 | 0.392 | 0.372 | 0.388 | 0.388 | 0.361| 0.320 | 0.285 |
| Traffic | 0.231 | 0.225 | 0.258 | 0.278 | 0.272 | 0.291| 0.287 | 0.305 |
| Weather | 0.231 | 0.225 | 0.258 | 0.276 | 0.272 | 0.291| 0.297 | 0.315 |

#### 5.5 The Effect of Deep Memory
In this section, we evaluate the effect of deep memory in both wall-clock training time and model performance^2. To this end, we focus on different variants of our neural memory module, where \( L_M = 1, 2, 3, 4 \). We also use Mamba as a baseline for the model performance. For a fair comparison, we use the same training process for all models and train them on a subset of the Pile dataset (L. Gao et al. 2020).

We report the perplexity of our models and baselines as the function of the sequence length in Figure 7. Interestingly, with the increase of memory depth, \( L_M \), the model can achieve better perplexity over all sequence length. Also, deeper memory modules are more robust to the sequence length when the model has less number of parameters. With the increase of the number of parameters, all models show better performance on longer sequences.

We also evaluate the effect of memory depth \( (L_M = 1, 2, 3, 4) \) on the training throughput. We report the training throughput (the number of tokens per second) as the function of sequence length in Figure 8. All models scale linearly with respect to the context length (i.e., constant trend in the number of tokens per second with respect to sequence length). Also, by increasing the memory depth, as expected, we can see a linear trend that a deeper memory results in a slower training. Therefore, it is not always efficient to use deeper memory modules, showing a trade-off between effectiveness and efficiency.

### Figure 8: The effect of memory depth on training throughput
![Memory depth effect](#)

#### 5.6 Time Series Forecasting
To show the effectiveness of our memory module in a broader tasks, we also evaluate its performance in time series forecasting tasks. To this end, we use the Simba framework (Patro and Agneeswaran 2024) for time series forecasting, and

| Model | Enhancer Cohn | Enhancer Ens | Human Reg | Non-TATA Promoters | Human OCR Ens |
|-----------------------|---------------|--------------|-----------|-------------------|----------------|
| CNN | 69.5 | 68.9 | 93.3 | 84.6 | 68.0 |
| DNABERT | 74.0 | 85.7 | 81.1 | 85.6 | 75.1 |
| GPT | 70.5 | 83.5 | 91.5 | 87.7 | 73.0 |
| HyenaDNA | 74.2 | 82.2 | 93.8 | 96.6 | 89.0 |
| Transformer++ | 73.4 | 89.5 | 89.9 | 94.4 | 79.5 |
| Mamba | 73.0 | - | - | 96.6 | - |
| Based | 74.6 | 89.5 | 89.5 | 96.8 | 79.0 |
| Neural Memory Module | 75.2 | 89.6 | 89.3 | 96.6 | 79.9 |

### 5.7 DNA Modeling
In order to understand the capability of Titans beyond natural language, we further evaluate the performance of our neural memory module on DNA modeling tasks. To this end, we evaluate pre-trained models on the downstream tasks in GenomicsBenchmarks (Grešóva et al. 2023). We follow the same experimental setups from Nguyen et al. (2024), and re-use the reported results of baselines by Arora et al. (2024). The performance of Titans (LMM) and baselines are reported in Table 4. We find that LMM is competitive with state-of-the-art architectures across different downstream genomics tasks.

### 5.8 Efficiency
In this part, we compare the efficiency of our neural memory as well as Titans with state-of-the-art sequence models. The training throughput of models for different sequence length × batch size are reported in Figure 9. Comparing recurrent models, including our neural memory module, we can see our memory module is slightly slower than Mamba2 and Gated DeltaNet, mainly due to: (1) having bigger memory and more expressive transition process (memory update), and (2) highly optimized kernel in the implementation of Mamba. Interestingly, Titans (MAL) are faster than baselines as well as the memory module. The main reason for this better throughput is the highly optimized kernel of Flash-Attention (Dao 2024), which is used for implementing SWA and full attention module in Titans.

### 5.9 Ablation Study
Finally, we perform ablation studies on the different architectural choices in Titans. We consider our neural memory module as a base model and then changing one component at a time: (1) replacing deep memory with linear memory, removing (2) convolution, (3) momentum in the surprise measure, (4) weight decay (or forgot mechanism), and (5) persistent memory. The results are reported in Table 5. All components of our neural memory design are positively contributing to its performance, where the greatest contribution comes from weight decay, momentum, convolution, and persistent memory, respectively.

The Effect of Architectural Design. To evaluate the effect of architecture design, we compare the performance of three represented variants of Titans in three aspects of (i) language modeling, (ii) common-sense reasoning, and (iii) long context NIAH (BABILon) tasks. The results are reported in Table 5. We find that MAC and MAG have close performance in language modeling and common-sense reasoning tasks, while MAC achieve significantly better performance in long-context NIAH. Both of these models achieve better performance than MAL. These results along with Figure 9, show a trade-off between fast training and more expressive design.
| Model | Language Modeling | Reasoning | Long Context |
|---------------------|-------------------|-----------|--------------|
| | ppl ↓ | acc ↑ | acc ↑ |
| LLM | 27.01 | 47.83 | 92.68 |
| +Attn (MAC) | 26.67 | 48.65 | 97.95 |
| +Attn (MAG) | 25.70 | 48.60 | 96.70 |
| +Attn (MAL) | 25.91 | 48.77 | 96.91 |
| Linear Memory | 28.49 | 46.97 | 85.34 |
| w/o Convolution | 28.73 | 45.82 | 90.28 |
| w/o Momentum | 28.98 | 45.49 | 87.12 |
| w/o Weight Decay | 29.04 | 45.11 | 85.60 |
| w/o Persistent Memory| 27.63 | 46.35 | 92.49 |

6 Conclusion

In this paper, we present a neural long-term memory that, as a meta in-context learner, learns to memorize at test time. The neural memory module is a recurrent model in nature, and is adaptively memorizing tokens that are more surprising or are close to surprising tokens. Comparing to modern recurrent models, it has more expressive memory update and storing mechanism. Using this memory, we present Titans architectures, and its three variants, in which we suggest to incorporate the memory module as (1) a context, (2) gating, and (3) a layer. Our experimental evaluation on diverse tasks validate that Titans are more effective than Transformers and recent modern linear recurrent models, specifically for long context. That is, Titans can scale to larger than 2M context window size with better accuracy than baselines.

Titans are implemented in Pytorch and JAX and we intend to make the code we use to train and evaluate our models available soon.
## References

[1] John Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diego Jiménez, Janiko Altschmidt, Sam Altman, Shyam Sundar Ankadak, et al. “Gpt-4 technical report”. In: arXiv preprint arXiv:2303.08774 (2023).

[2] Yaroslav Aksenov, Nikita Balaganskiy, Sofia Maria Lo Cicero Vaina, Boris Shaposhnikov, Alexey Gorbatovskiy, and Daniil Gavrilov. “Linear Transformers with Learnable Kernel Functions are Better In-Context Models”. In: arXiv preprint arXiv:2402.10644 (2024).

[3] Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando De Freitas. “Learning to learn by gradient descent by gradient descent”. In: Advances in neural information processing systems 29 (2016).

[4] Cem Anil, Yuhao Wu, Anders Andreassen, Aitor Lewkowycz, Vedant Mirsan, Ambrose Slone, Guy Gur-Ari, Ethan Dyer, and Behnam Neyshabur. “Exploring length generalization in large language models”. In: Advances in Neural Information Processing Systems 35 (2022), pp. 38546–38556.

[5] Simran Arora, Sabri Eyuboglu, Michael Zhang, Aman Timalsina, Silas Alberti, James Zou, Arti Rudra, and Christopher K. Pierre. “Single-head attention language models balance the recall-throughput tradeoff”. In: Forty-first International Conference on Machine Learning. 2024. URL: https://openreview.net/forum?id=e93fDcpH3.

[6] Dmitry Bahdanau. “Neural machine translation by jointly learning to align and translate”. In: arXiv preprint arXiv:1409.0473 (2014).

[7] Reza Bayat, Mohammad Pezeshki, Elvis Dhotmad, David Lopez-Paz, and Pascal Vincent. “The Pitfalls of Memorization: When Memorization Hurts Generalization”. In: arXiv preprint arXiv:2412.07684 (2024).

[8] Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Gunther Klambauer, Johannes Brandstetter, and Sep Höcherlreiter. “xLSTM: Extended Long Short-Term Memory”. In: arXiv preprint arXiv:2405.04517 (2024).

[9] Ali Behrouz, Michele Santacatterina, and Ramin Zabih. “Mambamixer: Efficient selective state space models with layer channel selection”. In: arXiv preprint arXiv:2403.19888 (2024).

[10] Yichen Chen, Eric Wendland, and Aijun Tan. “Like Zeroth, and Like Zettelmeyer, and Gargi Gosain: Memory Layers at Scale”. In: arXiv preprint arXiv:2412.09764 (2024).

[11] Albert Bittai, Vincent Calinescu, Diane Bouvier, Herve Jeeg, and Leon Bottou. “Birth of a transformer: a memory viewpoint”. In: Advances in Neural Information Processing Systems 36 (2024).

[12] Jindong Fan, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. “Piga: Reasoning about physical commonsense in natural language”. In: Proceedings of the AAAI conference on artificial intelligence. Vol. 34. 2020, pp. 7432–7439.

[13] Aleksandar Botev, Soham De, Samuel L Smith, Anushka Fernando, George-Cristian Muraru, Ruba Haroun, Leonard Barrada, Razvan Pascanu, Pier Giuseppe Essea, Robert Dadhia, et al. “RecurrentGemming Most Transformers for Efficient Open Language Models”. In: arXiv preprint arXiv:2404.07839 (2024).

[14] Léon Bottou and Vladimir Vapnik. “Local learning algorithms”. In: Neural computation 4.6 (1992), pp. 888–900.

[15] Aydar Bulatov, Yuri Kuratov, Veronik Kapushesv, and Mikhail S Burtsev. “Scaling transformer to lm tokens and beyond with rmt”. In: arXiv preprint arXiv:2304.11062 (2023).

[16] Aydar Bulatov, Yuri Kuratov, and Mikhail Burtsev. “Recurrent memory transformer”. In: Advances in Neural Information Processing Systems 35 (2022), pp. 11079–11091.

[17] Edoardo Cetin, Qi Sun, Tianyu Zhao, and Yujin Tang. “An Evolved Universal Transformer Memory”. In: arXiv preprint arXiv:2410.13160 (2024).

[18] Beidi Chen, Tri Dao, Eric Winsor, Zhao Song, Arti Rudra, and Christopher K. “Scatterbrain: Unifying sparse and low-rank attention”. In: Advances in Neural Information Processing Systems 34 (2021), pp. 17143–17126.

[19] Krzysztof Marcin Choromanski, Valeri Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afro Mohodhin, Lukasz Kaiser, David Benjamin Belanger, Lucy J Colwell, and Adrian Weller. “Rethinking Attention with Performers”. In: Intercontinental Conference on Learning Representations. 2021. URL: https://openreview.net/forum?id=U6Zk0W8R.hl.

[20] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. “BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions”. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). Ed. by Jill Burstein, Christy Doran, and Thamar Solorio. Minneapolis, Minnesota: Association for Computational Linguistics, June 2019, pp. 2924–2936. doi: 10.18653/v1/N19-1300. URL: https://aclanthology.org/N19-1300/.
## References

[1] John Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diego Jiménez, Janiko Altschmidt, Sam Altman, Shyam Sundar Ankadak, et al. “Gpt-4 technical report”. In: arXiv preprint arXiv:2303.08774 (2023).

[2] Yaroslav Aksenov, Nikita Balaganskiy, Sofia Maria Lo Cicero Vaina, Boris Shaposhnikov, Alexey Gorbatovskiy, and Daniil Gavrilov. “Linear Transformers with Learnable Kernel Functions are Better In-Context Models”. In: arXiv preprint arXiv:2402.10644 (2024).

[3] Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando De Freitas. “Learning to learn by gradient descent by gradient descent”. In: Advances in neural information processing systems 29 (2016).

[4] Cem Anil, Yuhao Wu, Anders Andreassen, Aitor Lewkowycz, Vedant Mirsan, Ambrose Slone, Guy Gur-Ari, Ethan Dyer, and Behnam Neyshabur. “Exploring length generalization in large language models”. In: Advances in Neural Information Processing Systems 35 (2022), pp. 38546–38556.

[5] Simran Arora, Sabri Eyuboglu, Michael Zhang, Aman Timalsina, Silas Alberti, James Zou, Arti Rudra, and Christopher K. Pierre. “Single-head attention language models balance the recall-throughput tradeoff”. In: Forty-first International Conference on Machine Learning. 2024. URL: https://openreview.net/forum?id=e93fDcpH3.

[6] Dmitry Bahdanau. “Neural machine translation by jointly learning to align and translate”. In: arXiv preprint arXiv:1409.0473 (2014).

[7] Reza Bayat, Mohammad Pezeshki, Elvis Dhotmad, David Lopez-Paz, and Pascal Vincent. “The Pitfalls of Memorization: When Memorization Hurts Generalization”. In: arXiv preprint arXiv:2412.07684 (2024).

[8] Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Gunther Klambauer, Johannes Brandstetter, and Sep Höcherlreiter. “xLSTM: Extended Long Short-Term Memory”. In: arXiv preprint arXiv:2405.04517 (2024).

[9] Ali Behrouz, Michele Santacatterina, and Ramin Zabih. “Mambamixer: Efficient selective state space models with layer channel selection”. In: arXiv preprint arXiv:2403.19888 (2024).

[10] Yichen Chen, Eric Wendland, and Aijun Tan. “Like Zeroth, and Like Zettelmeyer, and Gargi Gosain: Memory Layers at Scale”. In: arXiv preprint arXiv:2412.09764 (2024).

[11] Albert Bittai, Vincent Calinescu, Diane Bouvier, Herve Jeeg, and Leon Bottou. “Birth of a transformer: a memory viewpoint”. In: Advances in Neural Information Processing Systems 36 (2024).

[12] Jindong Fan, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. “Piga: Reasoning about physical commonsense in natural language”. In: Proceedings of the AAAI conference on artificial intelligence. Vol. 34. 2020, pp. 7432–7439.

[13] Aleksandar Botev, Soham De, Samuel L Smith, Anushka Fernando, George-Cristian Muraru, Ruba Haroun, Leonard Barrada, Razvan Pascanu, Pier Giuseppe Essea, Robert Dadhia, et al. “RecurrentGemming Most Transformers for Efficient Open Language Models”. In: arXiv preprint arXiv:2404.07839 (2024).

[14] Léon Bottou and Vladimir Vapnik. “Local learning algorithms”. In: Neural computation 4.6 (1992), pp. 888–900.

[15] Aydar Bulatov, Yuri Kuratov, Veronik Kapushesv, and Mikhail S Burtsev. “Scaling transformer to lm tokens and beyond with rmt”. In: arXiv preprint arXiv:2304.11062 (2023).

[16] Aydar Bulatov, Yuri Kuratov, and Mikhail Burtsev. “Recurrent memory transformer”. In: Advances in Neural Information Processing Systems 35 (2022), pp. 11079–11091.

[17] Edoardo Cetin, Qi Sun, Tianyu Zhao, and Yujin Tang. “An Evolved Universal Transformer Memory”. In: arXiv preprint arXiv:2410.13160 (2024).

[18] Beidi Chen, Tri Dao, Eric Winsor, Zhao Song, Arti Rudra, and Christopher K. “Scatterbrain: Unifying sparse and low-rank attention”. In: Advances in Neural Information Processing Systems 34 (2021), pp. 17143–17126.

[19] Krzysztof Marcin Choromanski, Valeri Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afro Mohodhin, Lukasz Kaiser, David Benjamin Belanger, Lucy J Colwell, and Adrian Weller. “Rethinking Attention with Performers”. In: Intercontinental Conference on Learning Representations. 2021. URL: https://openreview.net/forum?id=U6Zk0W8R.hl.

[20] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. “BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions”. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). Ed. by Jill Burstein, Christy Doran, and Thamar Solorio. Minneapolis, Minnesota: Association for Computational Linguistics, June 2019, pp. 2924–2936. doi: 10.18653/v1/N19-1300. URL: https://aclanthology.org/N19-1300/.
[21] Peter Clark, Isaac Cowley, Oren Etzioni, Tushar Koth, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. “Think you have solved question answering? try are, the ai2 reasoning challenge”. In: arXiv preprint arXiv:1803.05457 (2018).

[22] Nelson Cowan. “What are the differences between long-term, short-term, and working memory?” In: Progress in brain research 169 (2008), pp. 323–338.

[23] Zihang Dai, Zhihlin Yang, Yiming Yang, Jaime G. Carbonell, Quoc Viet Le, and Ruslan Salakhutdinov. “Transformer-XL: Attentive Language Models beyond a Fixed-Length Context”. In: ACL (1). Ed. by Anna Korhonen, David R. Traum, and Luis Marquez. Association for Computational Linguistics, 2019, pp. 2978–2988. ISBN: 978-1-950737-48-2.

[24] Tri Dao, “FlashAttention: 2 - Faster Attention with Better Parallelism and Work Partitioning”. In: The Twelfth International Conference on Learning Representations. 2024. URL: https://openreview.net/forum?id=mZNyh95Ec.

[25] Tri Dao, Dan Fu, Stefanie Emon, Ari Rudra, and Christopher Re. “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness”. In: Advances in Neural Information Processing Systems. Ed. by S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh. Vol. 35. Curran Associates, Inc., 2022, pp. 16344–16359. URL: https://proceedings.neurips.cc/paper_files/paper/2022/file/67673c320f20da302b1d36e4d5-Conference.pdf.

[26] Tri Dao and Albert Gu. “Transformers are S2Ms: Generalized models and efficient algorithms through structured state space duality”. In: arXiv preprint arXiv:2405.21600 (2024).

[27] Abhimanyu Das, Weizhao Kong, Andrew Leech, Shaan K. Mathur, Rajat Sen, and Rose Yu. “Long-Range Forecasting with TIDE: Time-series Dense Encoder”. In: Transactions on Machine Learning Research (2023). ISBN: 2535-8566. URL: https://openreview.net/forum?id=Ph2DcbQ3BMS.

[28] Soham De, Samuel L. Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Mihai, Albert Gu, Ruba Haroun, Leonard Beaudoin, Yutian Chen, Srivatsan Srinivasan, et al. “Griffin: Mixing gated linear recurrences with local attention for efficient language models”. In: arXiv preprint arXiv:2402.19427 (2024).

[29] Juechew Cheng, Boyuan Feng, Darius Gouskova, Yang Liang, and Horace He. “Flex Attention: A Programming Model for Generating Optimized Attention Kernels”. In: arXiv preprint arXiv:2412.05946 (2024).

[30] Matthyas Van Keirskbeek, Min-Hung Chen, Yoshi Suhara, et al. “Hymba: A Hybrid-head Architecture for Small Language Models”. In: arXiv preprint arXiv:2411.13676 (2024).

[31] Stefan Elfwine, Eiji Uchibe, and Kenji Doya. “Sigmoid-weighted linear units for neural network function approximation in reinforcement learning”. In: Neural networks 107 (2018), pp. 3–11.

[32] Yukun Feng, Feng Li, Ziang Song, Boyuan Zheng, and Philipp Koehn. “Learn to remember: Transformer with recurrent memory for document-level machine translation”. In: arXiv preprint arXiv:2210.01546 (2022).

[33] Daniel Y. Tri Dao, Khadkel Kamal Bashir M. Thomas, Ari Rudra, and Christopher Re. “Hungry Hungry Hippos: Towards Language Modeling with State Space Models”. In: The Eleventh International Conference on Learning Representations. 2023. URL: https://openreview.net/forum?id=2200Zy6W.

[34] Yossi Gandelman, Yu Sun, Xinlei Chen, and Alexei Efros. “test-time training with masked autoencoders”. In: Advances in Neural Information Processing Systems 35 (2022), pp. 29734–29385.

[35] Leo Gao, Stella Biderman, Sid Black, Laurenee Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Tihde, Noah Nabeshima, et al. “The aLx: an 8080 dataset for diverse text for language modeling”. In: arXiv preprint arXiv:2101.00027 (2020).

[36] Felix A. Gers, Jürgen Schmidhuber, and Fred Cummins. “Learning to forget: Continual prediction with LSTM”. In: Neural computation 12.10 (2000), pp. 2451–2471.

[37] Alex Graves, Greg Wayne, and Ivo Danihelka. Neural Turing Machines. 2014. arXiv: 1410.5401 [cs.NE]. URL: https://arxiv.org/abs/1410.5401.

[38] Klaus Greff, Ruqesh K Srivastava, Jan Koutník, Bas R Steunebrink, and Jürgen Schmidhuber. “LSTM: A search space Odyssey”. In: IEEE transactions on neural networks and learning systems 28.10 (2016), pp. 2222–2232.

[39] Katarína Grešová, Vitališt Martinek, David Čechák, Petr Šimeček, and Panagiotis Alexiou. “Genomic benchmarks: a collection of datasets for genomic sequence classification”. In: BMC Genomic Data 4.1 (2023), p. 25.

[40] Albert Gu and Tri Dao. “Mamba: Linear-Time Sequence Modeling with Selective State Spaces”. In: First Conference on Language Modeling. 2024. URL: https://openreview.net/forum?id=etSkWLjVA.

[41] Albert Gu, Karan Gehlot, and Christopher Re. “Efficiently Modeling Long Sequences with Structured State Spaces”. In: International Conference on Learning Representations. 2022. URL: https://openreview.net/forum?id=uVLfoZ1vAC.
[42] Chi Han, Qifan Wang, Hao Peng, Wenhan Xiong, Yu Chen, Heng Ji, and Sinong Wang. "LM-Infinite: Zero-Shot Extreme Length Generalization for Large Language Models". In: *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*. Ed. by Kevin Du, Helena Gome, and Steven Bethard. Mexico City, Mexico: Association for Computational Linguistics, June 2024, pp. 3991–4008. DOI: 10.18653/v1/2024.naacl-long.222. URL: https://aclanthology.org/2024.naacl-long.222.

[43] Ramin Hasani, Matthias Lechner, Tsun-Hsuan Wang, Makram Chaaban, Alexander Amini, and Daniela Rus. "Liquid Structural State-Space Models". In: *The Eleventh International Conference on Learning Representations*. 2023. URL: https://openreview.net/forum?id=40TkRKF57R.

[44] Zexue He, Leonid Kairyzin, Donghyun Kim, Julian McAuley, Dmitry Krotov, and Rogerio Ferreira. "CAMELot: Towards Large Language Models with Training-Free Consolidated Associative Memory". In: *arXiv preprint arXiv:2024.13449* (2024).

[45] Donald Olding Hebb. *The organization of behavior: A neuropsychological theory*. Psychology press, 2005.

[46] John J. Hopfield. "Neural networks and physical systems with emergent collective computational abilities." In: *Proceedings of the National Academy of Sciences 79.8* (1982), pp. 2554–2558.

[47] Kurt Hornik, Maxwell Stinchcombe, and Halbert White. "Multilayer feedforward networks are universal approximators". In: *Neural Networks 2.5* (1989), pp. 359–366.

[48] Cheng-Ping Hsieh, Siwei Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, and Boris Ginsburg. "RULER: What Is the Best Context Size for Long-Context Language Models?" In: *First Conference on Language Modeling*. 2022. URL: https://openreview.net/forum?id=IdKobc765Y.

[49] Delsey Hutchins, Imanol Salch, Yuhai Wu, Ethan Dyer, and Behman Neyshabur. "Block-recurrent transformers". In: *Advances in Neural Information Processing Systems 35* (2022), pp. 32348–33261.

[50] Kazuki Irie, Róbert Csordás, and Jürgen Schmidhuber. "The dual form of neural networks revisited: Connecting test time predictions to training patterns via spotlight of attention". In: *International Conference on Machine Learning*. PMLR. 2022, pp. 9639–9659.

[51] Kazuki Irie, Róbert Csordás, and Jürgen Schmidhuber. "Going deeper with transformers with recurrent fast weight programmers". In: *Advances in Neural Information Processing Systems 34* (2021), pp. 7703–7717.

[52] Vidit Jain and Erik Learned-Miller. "Online domain adaptation of a pre-trained cascade of classifiers". In: *CVPR 2021*. IEEE, 2011, pp. 577–584.

[53] Albert I. Ong, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. "Mistral 7B". In: *arXiv preprint arXiv:2306.06225* (2023).

[54] Praneeth Kacham, Vahab Mirrokni, and Peilin Zhong. "PolySketchFormer: Fast Transformers via Sketching Polynomial Kernels". In: *Forty-first International Conference on Machine Learning*. 2022. URL: https://openreview.net/forum?id=grHfdjFjK.

[55] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. "Scaling laws for neural language models". In: *arXiv preprint arXiv:2001.08361* (2020).

[56] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. "Transformers are NNs: fast autoregressive transformers with linear attention". In: *International Conference on Machine Learning*. PMLR. 2020, pp. 5156–5165.

[57] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. "Generalization through Memorization: Nearest Neighbor Language Models". In: *International Conference on Learning Representations*. 2020. URL: https://openreview.net/forum?id=HklBjdEKh.

[58] Yuri Kurov, Andrey Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Igorevich Sorokin, Artem Sorokin, and Mikhail Burtsev. "BABYLON: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack". In: *The Thirty-eighth Conference on Neural Information Processing Systems Datasets and Benchmarks Track*. 2024. URL: https://openreview.net/forum?id=u7m2c6gK2H.

[59] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. "Retrieval-augmented generation for knowledge-intensive NLP tasks". In: *Advances in Neural Information Processing Systems 33* (2020), pp. 9459–9474.
In this formulation, Gated DeltaNet is the same as above but with an additional weight decay term (S. Yang, Kautz, and Hataimatzi 2024). Comparing Equation 32 and Equation 34, we can see that setting η_t = 0 results in both formulations to be equivalent. Accordingly, we can say LMM is generalizing the very recent study of Gated DeltaNet (S. Yang, Kautz, and Hataimatzi 2024) from three aspects:

- **Momentum-based Rule:** The Delta Rule is based on momentary surprise, meaning that the flow of tokens cannot affect the memory update rule. LMM, however, is based on a momentum rule, which consider both past and momentary surprise.

- **Deep Memory:** While Gated DeltaNet is limited to a linear (matrix-valued) memory as it requires finding the closed recurrence form, LMM allows using deep memory module by using a gradient-based formulation, resulting in higher expressive power.

- **Non-Linear Recurrence:** While DeltaNet and Gated DeltaNet are based on linear recurrence, our LMM is using inter-chunk non-linear recurrence and intra-chunk linear recurrence. This design allows LMM having a higher expressive power.

Here, we discussed Gated DeltaNet as a sample of recent generation of recurrent models. Similar approaches such as RKW-7 (Peng 2021) are also using the same formulation and loss function, and so LMM is generalizing all such models.

**LMM is Generalized Longhorn.** Similar to DeltaNet, Longhorn (B. Liu et al. 2024) uses the same loss function but it derives the closed form using implicit online learning:

S_{t+1} = S_t \left( I - δ_t k_k^T \right) + δ_t v_k^T,

where δ_t = \frac{\partial}{\partial k_k^T}. It, however, lacks a forgetting gate, resulting in a faster memory overflow. Therefore, in addition two the aforementioned aspects of (1) Momentum-based Rule, (2) Deep Memory, and (3) Non-Linear Recurrence, LMM has the advantage of using an additional (4) Forget Gate, leading to a better memory management.

**LMM is Generalized TTT Layer.** To the best of our knowledge, TTT (Yu Sun et al. 2024), is the only modern linear recurrent models with a gradient-based updating rule. In addition to different architectural designs and also objective functions, our LMM has three key differences with presented TTT layers (Yu Sun et al. 2024):

1. **Forgetting Mechanism:** TTT layers are updating memory at each time, without having the chance to forget the past data. Accordingly, when fixing the memory size, the model cannot manage the memory for long sequences. A forget mechanism, such as LMM's, allows clearing the memory when very past information is not needed anymore. We show that in a general case, this forget mechanism is equivalent to weight decay and provide a fast method to incorporate it into the parallel training.

2. **Momentum-based Update Rule:** TTT layers are based on momentary surprise, meaning that the flow of tokens cannot affect the memory update rule. LMM, however, is based on a momentum rule, which consider both past and momentary surprise. See Section 3.1 for the motivation of this design.

3. **Deep Memory:** While TTT-layers allows for deeper memory, the advantages/disadvantages of such deeper memory modules have not been experimentally evaluated.

To the best of our knowledge, our neural long-term memory module is the first linear recurrent model with momentum-based update rule.

Finally, as a key difference with all the above and other recent linear recurrent studies, note that the hybrid variants of modern linear models—such as Griffin (De et al. 2024), DeltaNet (S. Yang, B. Wang, Yu Zhang, et al. 2024), Gated DeltaNet (S. Yang, Kautz, and Hataimatzi 2024), H3 (D. Y. Fu et al. 2023), Mamba2 (Dao and Gu 2024), Samba (Ren et al. 2024), etc.—all are based on sequential layer-wise design. We present Titans to show how effectively one can incorporate such memory modules into an architecture.