"""
Synthetic world for the trial study: 
Here we define:
	- Source
	- Environment
	- 3 test instruments
	- Events
"""

import numpy as np

PARAMS = dict(
phi_d = 0.80, sigma_d = 1.00,                  # source
phi_x = 0.70, beta = 0.50, sigma_x = 0.5,      # environment
L = 6, H = 6,                                  # source lag -> environment, forecast horizon
sigma_A = 0.50, sigma_B = 0.50, sigma_C = 2.0, # instrument noise / signal 
q_event = 0.97,                                # event threshold quantile
burn = 1000                                    # discarded starting steps
)

def simulate(T, params = PARAMS, seed = 0):
	"""
	returns a dict with d, x, a, b, c, event, y, theta threshold
	"""
	p = params
	rng = np.random.default_rng(seed)
	n = T + p['burn']
	
	# source d
	d = np.zeros(n)
	eta = rng.normal(size = n)
	for t in range (n - 1):
		d[t + 1] = p['phi_d'] * d[t] + p['sigma_d'] * eta[t]
		
	# environment x
	x = np.zeros(n)
	eps = rng.normal(size = n)
	for t in range (p['L'], n - 1):
		x[t + 1] = p['phi_x'] * x[t] + p['beta'] * d[t - p['L']] + p['sigma_x'] * eps[t]
		
	# discard initial transient
	d, x = d[p['burn']:], x[p['burn']:]
	
	# observation of the 3 instruments
	a = d + p['sigma_A'] * d.std() * rng.normal(size = T)
	b = x + p['sigma_B'] * x.std() * rng.normal(size = T)
	c = x + p['sigma_C'] * x.std() * rng.normal(size = T)
	
	# defining threshold and event
	theta = np.quantile(x, p['q_event'])
	event = (x > theta).astype(int) 
	
	# label to be forecasted 
	y = np.zeros(T, dtype = int)
	for t in range(T-p['H']):
		y[t] = int(event[t + 1 : t + p['H'] + 1].any())
			
	valid = np.arange(T) < T - p['H']
	
	return dict(d = d, x = x, a = a, b = b, c = c, event = event, y = y, valid = valid, theta = theta)
	
if __name__ == "__main__":
	w = simulate(100_000)
	print(f"theta = {w['theta']:.2f}")
	print(f"event fraction = {w['event'].mean():.3f}")
	print(f"fraction of y = 1 = {w['y'][w['valid']].mean():.3f}")
