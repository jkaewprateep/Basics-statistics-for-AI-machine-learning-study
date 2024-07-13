# Basics-statistics-for-AI-machine-learning-study
Statistics with Python Specialization [MICHIGAN]( https://coursera.org/share/0c7445ac88c5b05923f937afdd64c925 ) 🔶🔷 </br>
AI Developer Professional [IBM]( https://coursera.org/share/95fa5c2bf36ea52759dcabc50e1a81b0 ) 🔶🔷 </br>
Data Analytics [Google]( https://coursera.org/share/3d15025b54bd5680458942a2d4e7c1a6 ) 🔶🔷 </br>

<p align="center" width="100%">
    <img alt="flappy_distance.jpg" width="34%" src="https://github.com/jkaewprateep/Basics-statistics-for-AI-machine-learning-study/blob/main/Screenshot%202024-07-14%20010127.png">
    <img alt="flappy_distance.jpg" width="14.2%" src="https://github.com/jkaewprateep/Basics-statistics-for-AI-machine-learning-study/blob/main/08419ff9-9066-4114-9af4-cca209abc322.jpg"> </br>       <b>Picture from Internet</b></br>
</p>

## Bayesian probability ##

<p align="center" width="100%">
    <img alt="flappy_distance.jpg" width="33%" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/flappy_distance.jpg">
    <img alt="flappy_distance.jpg" width="40%" src="https://github.com/jkaewprateep/lessonfrom_Applied_Plotting_Charting_and_Data_Representation_in_Python/blob/main/FlappyBird_small.gif"> </br>
    <b>Continuous graph | Sample application</b></br>
</p>

### Neuron networks sample codes ###

```
class MyDenseLayer(tf.keras.layers.Layer):
	def __init__(self, num_outputs, name="MyDenseLayer"):
		super(MyDenseLayer, self).__init__()
		self.num_outputs = num_outputs

	def build(self, input_shape):
		self.kernel = self.add_weight("kernel",
		shape=[int(input_shape[-1]), 1])
		self.biases = tf.zeros([int(input_shape[-1]), 1])

	def call(self, inputs):
	
		# Weights from learning effects with input.
		temp = tf.reshape( inputs, shape=(10, 1) )
		temp = tf.matmul( inputs, self.kernel ) + self.biases
		
		# Posibility of x in all and x.
		return tf.nn.softmax( temp, axis=0 )
```

<p align="center" width="200%">    
    $p(\theta|D) = P(D|\theta)P(\theta)/P(D)$ 
</p>



🧸💬 **The probability of event $\theta$ created after event $D$ is similar to event $D$ from $\theta$ and the probability of event $\theta$**. Similar to likelihood sequences when the first order in the sequence is $\theta$ and the next is $D$ the probability of $\theta$ and $D$ is less than $\theta$ only and we can manipulate the value with target probabilities to perform some processes such as comparing sequence likelihood, find sources original, create greeting response number from input sequence number and more ...

[Neuron-Networks-review]( https://github.com/jkaewprateep/lessonfrom_Applied_Plotting_Charting_and_Data_Representation_in_Python ) </br>
[Applied Plotting]( https://github.com/jkaewprateep/lessonfrom_Applied_Plotting_Charting_and_Data_Representation_in_Python )
</br> 

<p align="center" width="100%">
    <img alt="Statistics distribution" width="40%" src="https://github.com/jkaewprateep/Basics-statistics-for-AI-machine-learning-study/blob/main/picture_01.png">
    <img alt="Statistics distribution" width="40%" src="https://github.com/jkaewprateep/Basics-statistics-for-AI-machine-learning-study/blob/main/picture_03.png"> </br>
    <b>Picture from Internet</b>
</p>

### Data generator ###

```
def lf_plot(n):
    theta = np.linspace(0.01, 0.99, 100)
    for x in np.floor(np.r_[0.2, 0.5, 0.6]*n):
        l = st.binom.pmf(x, n, theta)
        plt.grid(True)
        plt.plot(theta, l, "-", label="%.0f" % x)
        plt.xlabel(r"$\theta$", size=15)
        plt.ylabel("Log likelihood", size=15)
    ha, lb = plt.gca().get_legend_handles_labels()
    plt.figlegend(ha, lb, "center right")
```

🧸💬 BETA priors is using for the function characteristics in example $\alpha = \beta = 1$ is center distribution, $\alpha, \beta > 1$ is mode, $\alpha, \beta < 1$  is anti-mode, mean, robustness, concentration and variance are use for technically graphs distribution performance.

<p align="center" width="100%">
    <img alt="Statistics distribution" width="40%" src="https://github.com/jkaewprateep/lessonfrom_Applied_Plotting_Charting_and_Data_Representation_in_Python/blob/main/02.png">
    <img alt="Statistics distribution" width="32%" src="https://github.com/jkaewprateep/Basics-statistics-for-AI-machine-learning-study/blob/main/picture_02.png"> </br>
    <b>Continuous graph | Confusion matrix</b></br>
    <b>Picture from Internet</b>
</p>
