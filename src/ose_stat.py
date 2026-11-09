"""
Statistical (non-assimilative) OSE: a classifier learns the event probability from lagged observations
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ose_assim import tss_max
from world import PARAMS, simulate

# statistical forecast engines: each entry builds a new, untrained model
LEARNERS = dict(
	logistic = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter = 1000)),
	boosting = lambda: HistGradientBoostingClassifier(random_state = 0),
)


def features(w, instruments, lags):
	"""
	One column per instrument and lag k = 0,...,lags , holding the measurement at t - k (Nan where t-k<0)
	"""

	T = len(w['x'])
	cols = []
	for name in instruments:
		for k in range(lags + 1):
			col = np.full(T, np.nan)
			col[k:] = w[name][:T - k]
			cols.append(col)

	return np.column_stack(cols)

def split(w, frac = 0.5):
	"""
	Training steps (first part of the series) and test steps (the rest)
	"""
	T = len(w['x'])
	first = np.arange(T) < frac * T
	return w['valid'] & first, w['valid'] & ~first

def stat_forecast(w, instruments, train, learner = 'logistic', lags = 12, target = 'y'):
	"""
	Probability of the target label at every step, learned from the catalogue on the training steps
	(target = 'y': event in the next H steps; target = 'event': event now)
	"""
	X = features(w, instruments, lags)
	ok = ~np.isnan(X).any(axis = 1)
	model = LEARNERS[learner]()
	model.fit(X[train & ok], w[target][train & ok])
	prob = np.full(len(X), np.nan)
	prob[ok] = model.predict_proba(X[ok])[:, 1]
	return prob

if __name__ == '__main__':
	w = simulate(100_000, PARAMS)
	train, test = split(w)
	for learner in LEARNERS:
		for inst in [('a','b','c'), ('b','c'), ('a','c'), ('a','b')]:
			prob = stat_forecast(w, inst, train, learner)
			print(learner, inst, 'TSS:', round(tss_max(w['y'][test], prob[test]), 3))
