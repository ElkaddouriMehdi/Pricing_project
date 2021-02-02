# Pricing_project
**Author** : ELKADDOURI MEHDI

### Summary  

Within the framework of this project, we will exploit the formula of black and scholes to price the call and the put during times and at maturity.   Similarly, we will use the simulation of Mont-carlo to price the European, asian,barrier calls and puts.  

On the other hand, we'll price the implied volatility using market data. For the stochastic volatility, we will use Heston model

***
### Packages and modules 

***my_data*** : it's a module of the package ***data*** that aims to manipulate data.  
***my_pricing*** : it's module that aims to price the data.  
***payoff*** : it calculates *the payoff* of differents products ( used in montecarlo simulation)  
***
 *each code is commented to make the user understand its functionnality* 

#### 2. pricing data using *the my_pricing* functions:
 ```math

\begin{equation}
r S_{t} \frac{\partial C}{\partial S}+\frac{\partial C}{\partial t}+\frac{1}{2} \sigma^{2} S^{2} \frac{\partial^{2} C}{\partial S^{2}}-r C=0
\end{equation}

is a Partial Differential Equation (PDE) describing how the option value $V(S, t)$ depends on the stock price $S$ and time $t$.
In equation above, $\sigma$ is the volatility of the underlying asset,and $r$ is the interest rate. Both $\sigma$ and $r$ are considered given, while $C$ is the quantity being computed (or approximated).

The call using ***black scholes formula***:
\begin{equation}
	\mathrm C(\mathrm S,\mathrm t)= \mathrm N(\mathrm d_1)\mathrm S - \mathrm N(\mathrm d_2) \mathrm K \mathrm e^{-rt}
	\label{eq:2}
\end{equation}
The put formula using ***Black scholes formula***:
\begin{equation}
	\mathrm P(\mathrm S,\mathrm t)= \mathrm -N(\mathrm -d_1)\mathrm S + \mathrm N(\mathrm -d_2) \mathrm K \mathrm e^{-rt}
	\label{eq:3}
\end{equation}  

where  

\begin{equation}
	\mathrm d_1= \frac{1}{\sigma \sqrt{\mathrm t}} \left[\ln{\left(\frac{S}{K}\right)} + t\left(r + \frac{\sigma^2}{2} \right) \right]
\end{equation}

\begin{equation}
	\mathrm d_2= \frac{1}{\sigma \sqrt{\mathrm t}} \left[\ln{\left(\frac{S}{K}\right)} + t\left(r - \frac{\sigma^2}{2} \right) \right]
\end{equation} 
***
```


