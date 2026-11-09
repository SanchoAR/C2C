"""
Assimilative OSE: Kalman filter on the synthetic world, with and without each instrument
"""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_curve
from world import PARAMS, simulate

def build_model(p, lags = 0):
	"""
	Transition matrix F and process noise Q for the augmented state s = [x, d, d-1, ... , d-L]
	(more past values of d if lags > L, for instruments that average over several steps)
	"""
	L = p['L']
	n = max(L, lags) + 2
	F = np.zeros((n,n))
	F[0,0] = p['phi_x']
	F[0,1+L] = p['beta']
	F[1,1] = p['phi_d']
	for j in range (1, n-1):
		F[j+1, j] = 1
	
	Q = np.zeros((n,n))
	Q[0,0], Q[1,1] = p['sigma_x']**2, p['sigma_d']**2
	
	return F, Q

# What each instrument sees:

OBS = dict(
a = (1, 'd', 'sigma_A'),
b = (0, 'x', 'sigma_B'),
c = (0, 'x', 'sigma_C')
)

def build_obs(p, w, instruments):
	"""
	Observation matrix Hm, noise covariance R, and observation z for the chosen instruments
	"""
	n = p['L'] + 2
	k = len(instruments)
	Hm = np.zeros((k, n))
	R = np.zeros((k,k))
	for i, name in enumerate(instruments):
		idx, qty, sig = OBS[name]
		Hm[i,idx] = 1
		R[i, i] = (p[sig] * w[qty].std()) ** 2
	z = np.column_stack([w[name] for name in instruments])
	return Hm, R, z
	
def kalman(F, Q, Hm, R, z):
	"""
	Kalman filter: analysed state at each step (one row per step) and final error covariance
	"""
	
	T, n = len(z), F.shape[0]
	s = np.zeros(n)
	P = np.eye(n) * 10.0
	I = np.eye(n)
	S = np.zeros((T,n))
	for t in range(T):
		#forecast
		s = F@s 
		P = F@P@F.T + Q
		
		#update
		K = P@Hm.T@np.linalg.inv(Hm@P@Hm.T + R)
		s = s + K @ (z[t] - Hm@s)
		P = (I - K @ Hm) @ P
		S[t] = s
	
	return S, P
	
def future_cov(F, Q, P, H):
	"""
	Covariance of the deviations of x[t+1], ..., x[t+H] from their expected values (the same for every t)
	"""
	C = P
	covs = []
	for h in range(H):
		C = F@C@F.T + Q
		covs.append(C)
	Sigma = np.zeros((H, H))
	for h in range(H):
		G = covs[h]										# cov(state[t+h+1]); cov(state[t+k+1], state[t+h+1]) = F^(k-h) cov(state[t+h+1])
		for k in range(h, H):
			Sigma[h, k] = Sigma[k, h] = G[0, 0]
			G = F@G
	return Sigma

def event_probability(S, F, Sigma, theta, M = 1000, seed = 1, chunk = 500):
	"""
	Probability that x exceeds theta at least once in t+1, ..., t+H, for every t.
	The futures are shifted towards the most likely exceedance and reweighted (importance sampling),
	so that the relative error stays small also for very small probabilities.
	"""
	T, H = len(S), len(Sigma)
	mean = np.zeros((T, H))
	X = S
	for h in range(H):
		X = X @ F.T
		mean[:, h] = X[:, 0]
	gap = theta - mean										# distance of each expected value from the threshold
	sd = np.sqrt(np.diag(Sigma))
	hstar = np.argmin(gap / sd, axis = 1)					# horizon where an exceedance is most likely
	push = np.maximum(gap[np.arange(T), hstar], 0)			# no shift if the expected path already exceeds theta
	shift = Sigma[:, hstar].T / Sigma[hstar, hstar][:, None] * push[:, None]	# most likely path that just reaches theta
	a = np.linalg.solve(Sigma, shift.T).T					# Sigma^-1 shift, one row per step
	rng = np.random.default_rng(seed)
	E = rng.standard_normal((M, H)) @ np.linalg.cholesky(Sigma).T		# unshifted futures, N(0, Sigma)
	prob = np.zeros(T)
	for i in range(0, T, chunk):
		sl = slice(i, i + chunk)
		hit = (E[None, :, :] > (gap[sl] - shift[sl])[:, None, :]).any(axis = 2)
		logw = -(a[sl] @ E.T) - 0.5 * (shift[sl] * a[sl]).sum(axis = 1)[:, None]	# likelihood ratio of the shifted futures
		prob[sl] = (hit * np.exp(logw)).mean(axis = 1)
	return prob
	
def tss_max(y, prob):
	"""
	Best TSS over all probability thresholds: max(hit rate- false alarm rate)
	"""
	fpr, tpr, _ = roc_curve(y, prob)
	return np.max(tpr-fpr)
	
def assim_forecast(w, instruments, train, model = PARAMS):
	"""
	Event probability at every step, from the Kalman filter run with the given model
	of the world (train is not used: filter does not learn from the catalogue)
	"""
	F, Q = build_model(model)
	Hm, R, z = build_obs(model, w, instruments)
	S, P = kalman(F, Q, Hm, R, z)
	Sigma = future_cov(F, Q, P, model['H'])
	return event_probability(S, F, Sigma, w['theta'])

def assim_nowcast(w, instruments, train, model = PARAMS):
	"""
	Probability that x is above theta now, from the Kalman analysis (gaussian error with variance P[0, 0])
	"""
	F, Q = build_model(model)
	Hm, R, z = build_obs(model, w, instruments)
	S, P = kalman(F, Q, Hm, R, z)
	return norm.sf((w['theta'] - S[:, 0]) / np.sqrt(P[0, 0]))

def assim_forecast_instrument(w, a, area, k, top, sigma_ref, model = PARAMS, where = None):
	"""
	Event probability from the Kalman filter fed with a synthetic configuration of A (instrument of run_3):
	every k steps the mean of d over the previous k steps, noise sigma_ref / sqrt(area * k), saturated at top.
	A saturated value only says that the measurement is at least top: censored (Tobit) update.
	where: steps where the probability is needed (all if None)
	"""
	F, Q = build_model(model, lags = k)
	n = F.shape[0]
	h = np.zeros(n)
	h[2:2+k] = 1 / k								# mean of d[t-1], ..., d[t-k]
	r = sigma_ref**2 / (area * k)
	P = np.eye(n) * 10.0
	phase_P = [None] * k							# steady-state covariance at each step of the measurement cycle
	for t in range(50 * k):
		P = F@P@F.T + Q
		if t % k == 0:
			K = P@h / (h@P@h + r)
			P = P - np.outer(K, h@P)
		phase_P[t % k] = P
	T = len(a)
	s = np.zeros(n)
	P = np.eye(n) * 10.0
	S = np.zeros((T, n))
	for t in range(T):
		s = F@s
		P = F@P@F.T + Q
		if t >= k and t % k == 0:					# a new block value arrives
			mu, var = h@s, h@P@h + r
			K = P@h / var
			if a[t] < top:
				m, v = a[t], 0.0
			else:									# saturated: moments of the predicted measurement above top
				alpha = (top - mu) / np.sqrt(var)
				lam = norm.pdf(alpha) / max(norm.sf(alpha), 1e-300)
				m, v = mu + np.sqrt(var) * lam, var * (1 + alpha * lam - lam**2)
			s = s + K * (m - mu)
			P = P - np.outer(K, K) * (var - v)
		S[t] = s
	where = np.ones(T, bool) if where is None else where
	prob = np.full(T, np.nan)
	for ph in range(k):
		sel = where & (np.arange(T) % k == ph)
		Sigma = future_cov(F, Q, phase_P[ph], model['H'])
		prob[sel] = event_probability(S[sel], F, Sigma, w['theta'])
	return prob

def true_state(w, L):
	"""
	Augmented state build from the true d and x, one row per step (first L steps skipped)
	"""
	
	T = len(w['x'])
	cols = [w['x'][L:], w['d'][L:]] + [w['d'][L-k:T-k] for k in range(1, L+1)]
	return np.column_stack(cols)
	
if __name__ == '__main__':
	p = PARAMS
	w = simulate(100_000, p)
	T = len(w['x'])
	test = w['valid'] & (np.arange(T) >= T/2)
	models = dict(perfect = PARAMS,
	wrong_beta = dict(PARAMS, beta = 0.3),
	wrong_lag = dict(PARAMS, L = 3))
	
	for name, model in models.items():
		for inst in [('a','b','c'),('b','c'), ('a','c'), ('a','b')]:
			prob = assim_forecast(w, inst, None, model)
			print(name, inst, 'TSS:', round(tss_max(w['y'][test], prob[test]),3),
			'mean probability:', round(prob[test].mean(), 3),
			'event rate:', round(w['y'][test].mean(), 3))
