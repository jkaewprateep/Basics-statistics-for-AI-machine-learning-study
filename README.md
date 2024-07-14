# Basics-statistics-for-AI-machine-learning-study
Statistics with Python Specialization [MICHIGAN]( https://coursera.org/share/0c7445ac88c5b05923f937afdd64c925 ) 🔶🔷 </br>
AI Developer Professional [IBM]( https://coursera.org/share/95fa5c2bf36ea52759dcabc50e1a81b0 ) 🔶🔷 </br>
Data Analytics [Google]( https://coursera.org/share/3d15025b54bd5680458942a2d4e7c1a6 ) 🔶🔷 </br>

<p align="center" width="100%">
    <img alt="flappy_distance.jpg" width="34%" src="https://github.com/jkaewprateep/Basics-statistics-for-AI-machine-learning-study/blob/main/Screenshot%202024-07-14%20010127.png">
    <img alt="flappy_distance.jpg" width="14.2%" src="https://github.com/jkaewprateep/Basics-statistics-for-AI-machine-learning-study/blob/main/08419ff9-9066-4114-9af4-cca209abc322.jpg"> </br>       <b>Picture from Internet</b></br>
</p>

## Bayesian probability ##
🧸💬 In the event there are some events before the current and continue, aligned of all the events created of our as humans nowadays. </br>
🐑💬 ➰ They can have multiple events created we need to find the streams and work for each stream as in the application of Speech recognition, NLP, image processing for multiple layers of object detection and augmentation reality, etc. We can have multiple agents queue for communications problems, it is called communications problems because they are updated and comparing results for target results and feedback propagation for the learning process. In multiple agent queues, there are internal updates, communication updates, and experience learning we call scenarios. [Multiple agent queues responses]( https://github.com/jkaewprateep/multi_agent_queue_action_response ) </br>
🦭💬 Most processes are individual records meaning multiple records are processed at once by selection but each record is performed for each process with a reference number, there is another approach to aggregate data records that supports multiple records performed from their results. They need reference for logging and verification methods, aggregate tables have specific methods for reference data management such as hash records or rollback compatibility methods. Moreover, some databases need to retain detailed tables with reference to target aggregate data for the verification process too. ( Some processes they need to update results or re-process need to maintain this table data over the retention period ) </br>
🐐💬 Process from designs can be validated and there is an event listener or dataflow diagram as a sample of the Markov model of the data application process because we can work with the state messages in the correct order. Incorrect priority message filters by communication method they need to update and verify of the event status and result, the state of the message similar to the Markov model as an example. [Markov chain model example]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/markov-chain.jpg )</br>

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

🧸💬 **The probability of event $\theta$ created after event $D$ is similar to event $D$ from $\theta$ and the probability of event $\theta$**. Similar to likelihood sequences when the first order in the sequence is $\theta$ and the next is $D$ the probability of $\theta$ and $D$ is less than $\theta$ only and we can manipulate the value with target probabilities to perform some processes such as comparing sequence likelihood, find sources original, create greeting response number from input sequence number and more ...

<p align="center" width="200%">    
    $p(\theta|D) = P(D|\theta)P(\theta)/P(D)$ 
</p>

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

🧸💬 Transforming data information is one of data identical identity, the application of AI machine learning finds patterns that can be used to categorize data, learn, and provide feedback. Marginal chatbox can consider multi-level linear regression problems. In marginal identification internal interceptions/randoms interception is study scopes data response for the function and when x-y interceptions are points of graph where x and y exist then the internal interceptions is a response to one function with existing data values. </br>

<p align="center" width="100%">
    <img alt="Statistics distribution" width="40%" src="https://github.com/jkaewprateep/Basics-statistics-for-AI-machine-learning-study/blob/main/picture_05.png">
    <img alt="Statistics distribution" width="40%" src="https://github.com/jkaewprateep/Basics-statistics-for-AI-machine-learning-study/blob/main/picture_04.png"> </br>
    <b>Data continuous | Marginal chatbox</b></br>
    <b>Picture from Internet</b>
</p>

## OLS - Ordinary Least Squares ##
🧸💬 One method applicable is ordinary least squares, work results, and experiment values easily applied by this method and perform the update with one or two variables concatenation for the robustness of variable input variances problems. </br>   

<p align="center" width="100%">
    <img alt="Statistics distribution" width="30%" src="https://github.com/jkaewprateep/Basics-statistics-for-AI-machine-learning-study/blob/main/picture_06.png">
    <img alt="Statistics distribution" width="50%" src="https://github.com/jkaewprateep/model_stability/blob/main/Figure_5.png"> </br>
    <b>Continuous BMXBMI | Continuous Flappy bird output variables </b></br>
    <b>Picture from Internet</b>
</p>

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 BPXSY1   R-squared:                       0.215
Model:                            OLS   Adj. R-squared:                  0.214
Method:                 Least Squares   F-statistic:                     697.4
Date:                Wed, 19 Jun 2024   Prob (F-statistic):          1.87e-268
Time:                        06:58:42   Log-Likelihood:                -21505.
No. Observations:                5102   AIC:                         4.302e+04
Df Residuals:                    5099   BIC:                         4.304e+04
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
=====================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------
Intercept           100.6305      0.712    141.257      0.000      99.234     102.027
RIAGENDRx[T.Male]     3.2322      0.459      7.040      0.000       2.332       4.132
RIDAGEYR              0.4739      0.013     36.518      0.000       0.448       0.499
==============================================================================
Omnibus:                      706.732   Durbin-Watson:                   2.036
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1582.730
Skew:                           0.818   Prob(JB):                         0.00
Kurtosis:                       5.184   Cond. No.                         168.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

```

## R-squared and correlation ##
🧸💬 In the first step we should create a communication method when working on a project, how much of the selected data variables are related to another or target variable output value? How much they are related to the objective variable values and response function output and how much it effects after applying some value or variables. Variable correlation is one good method to identify first. </br>

**Input:**
```
cc = da[["BPXSY1", "RIDAGEYR"]].corr()
print(cc.BPXSY1.RIDAGEYR**2)
```

**Output:**
```
0.2071545962518702
```

## Residual plot and errors band ##

🧸💬 From data observation, this should be done step by data visualization we found that the BMI value matches linear regression with the ordinary least squares model and error bands are in the finite values. </br>   

<p align="center" width="100%">
    <img alt="Statistics distribution" width="40%" src="https://github.com/jkaewprateep/Basics-statistics-for-AI-machine-learning-study/blob/main/picture_07.png">
    <img alt="Statistics distribution" width="40%" src="https://github.com/jkaewprateep/Basics-statistics-for-AI-machine-learning-study/blob/main/picture_08.png"> </br>
    <b>BMI linear plot | BMI scatter plot </b></br>
    <b>Picture from Internet</b>
</p>

## Logistics regression ##

🧸💬 In some application events had persistence scopes $p/(1-p)$ or $p1 = 1 - p2$ . This step is selecting a suitable model for the study and in our example are linear regression model and logistic regression model. </br>

<p align="center" width="100%">
    <img alt="Statistics distribution" width="30%" src="https://github.com/jkaewprateep/Basics-statistics-for-AI-machine-learning-study/blob/main/picture_09.png">
    <img alt="Statistics distribution" width="45%" src="https://github.com/jkaewprateep/counter_clocks/blob/main/98.png"> </br>
    <b>BMI Odd ( logistic regression ) | Sample application in devices </b></br>
    <b>Picture from Internet</b>
</p>

### Sample codes ###
```
temp = tf.random.normal([10], 1, 0.2, tf.float32)
temp = np.asarray(temp) * np.asarray([ coefficient_0, coefficient_1, coefficient_2, coefficient_3, 
coefficient_4, coefficient_5, coefficient_6, coefficient_7, coefficient_8, coefficient_9 ])
action = np.argmax(temp)	
```

### Sample target actions ###
```
actions = { "none_1": K_h, "up_1": K_w, "none_2": K_h, "none_3": K_h, "none_4": K_h, 
                      "none_5": K_h, "none_6": K_h, "none_7": K_h, "none_8": K_h, "none_9": K_h }
```

---

<p align="center" width="100%">
    <img width="30%" src="https://github.com/jkaewprateep/advanced_mysql_topics_notes/blob/main/custom_dataset.png">
    <img width="30%" src="https://github.com/jkaewprateep/advanced_mysql_topics_notes/blob/main/custom_dataset_2.png"> </br>
    <b> 🥺💬 รับจ้างเขียน functions </b> </br>
</p>
