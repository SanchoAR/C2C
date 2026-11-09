"""
Step 1d trial study (low contrast): does the method still find the best instrument class when two classes are worth almost
the same, and does it say so when they cannot be told apart?
The contrast between A and B is lowered by making A noisier (sigma_A); the same noise is given to the Kalman filter.
Step 1a: standard = class with the larger loss, always; ours = paired block bootstrap, A wins if P(loss A > loss B) >= 90%,
B wins if <= 10%, otherwise "indistinguishable".
Step 1c: standard MCDA (one weight) vs SMAA with random weights only vs SMAA with random weights and bootstrap criteria.
Truth: Kalman filter with the perfect model, mean over the worlds on the longest catalogue.

python3 run_1d.py            uses the saved raw results if they exist
python3 run_1d.py --rerun    recomputes everything (about 40 minutes on 8 cores)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ose_assim import assim_forecast, assim_nowcast
from ose_stat import split, stat_forecast
from run_4 import weights, tss_weighted
from world import PARAMS, simulate

RES = Path(__file__).parent.parent / 'results'
RAW = RES / 'run_1d_raw.csv'

SIGMAS = [0.5, 1.5, 2, 2.5, 3, 3.5, 4, 6]					# noise of A: the contrast A / B falls from about 30 to below 1
SEEDS = range(100, 130)										# independent worlds, as in step 4
LENGTHS = [50_000, 10_000, 2_000]							# test steps of the catalogue (2,000 steps: about 18 episodes)
BLOCK = 50													# block length of the bootstrap (longer than an episode, step 4)
DECIDE = 0.90												# a class wins only if it is ahead in at least this share of replicates
W_NOMINAL = 0.5												# weight of the forecast criterion in the standard MCDA
K_WEIGHTS = 100												# random weights per bootstrap replicate (SMAA with uncertain criteria)
WGRID = np.linspace(0, 1, 2001)								# forecast weights of the SMAA (two criteria: uniform on the simplex)
INSTRUMENTS = ('a', 'b', 'c')

# forecasts needed for each world: data denial (step 1a) and each instrument alone, forecast and nowcast (step 1c)
RUNS = {'all': ('a', 'b', 'c'), 'without a': ('b', 'c'), 'without b': ('a', 'c'),
        'only a': ('a',), 'only b': ('b',), 'only c': ('c',)}

def forecasts(w, train, p):
	"""Event probabilities of every run, for both engines: forecast (label y) and, for each instrument alone, nowcast (label event)."""
	out = dict(kalman = {}, boosting = {})
	for label, inst in RUNS.items():
		out['kalman'][label] = assim_forecast(w, inst, train, p)
		out['boosting'][label] = stat_forecast(w, inst, train, 'boosting', target = 'y')
	for i in INSTRUMENTS:
		out['kalman'][f'now {i}'] = assim_nowcast(w, (i,), train, p)
		out['boosting'][f'now {i}'] = stat_forecast(w, (i,), train, 'boosting', target = 'event')
	return out

def first_rank(U, wts):
	"""Share of the forecast weights wts that put each instrument first; U: instrument x criterion (forecast, nowcast)."""
	scores = wts[:, None] * U[:, 0] + (1 - wts[:, None]) * U[:, 1]
	best = scores.argmax(axis = 1)
	return np.array([(best == i).mean() for i in range(len(INSTRUMENTS))])

def run_world(seed, sigma):
	"""One world with a given noise of A: point values and bootstrap results for every catalogue length and engine."""
	p = dict(PARAMS, sigma_A = sigma)
	w = simulate(100_000, p, seed)
	train, test = split(w)
	idx = np.flatnonzero(test)
	probs = forecasts(w, train, p)
	y, event = w['y'][idx], w['event'][idx]
	rng = np.random.default_rng(seed)
	rows = []
	for n in LENGTHS:
		yn, en = y[:n], event[:n]
		m = len(yn)
		W = weights(m, BLOCK, rng)							# the same replicates for every run and engine (paired bootstrap)
		W = W[(W @ yn > 0) & (W @ (1 - yn) > 0) & (W @ en > 0) & (W @ (1 - en) > 0)]	# drop replicates without events
		W = np.vstack([np.ones(m, np.float32), W])			# row 0: the catalogue itself
		for engine, pe in probs.items():
			tss = {label: tss_weighted(yn, pe[label][idx][:n], W) for label in RUNS}
			now = {i: tss_weighted(en, pe[f'now {i}'][idx][:n], W) for i in INSTRUMENTS}
			loss_a, loss_b = tss['all'] - tss['without a'], tss['all'] - tss['without b']
			U = np.stack([np.stack([tss[f'only {i}'], now[i]], axis = -1) for i in INSTRUMENTS], axis = 1)	# replicate x instrument x criterion
			acc_w = first_rank(U[0], WGRID)							# SMAA, random weights only
			wr = rng.uniform(size = (len(W) - 1, K_WEIGHTS))		# SMAA, random weights and bootstrap criteria
			scores = wr[:, :, None] * U[1:, None, :, 0] + (1 - wr[:, :, None]) * U[1:, None, :, 1]
			best = scores.argmax(axis = -1)
			acc_wc = [(best == i).mean() for i in range(len(INSTRUMENTS))]
			row = dict(seed = seed, sigma_A = sigma, engine = engine, n = n, replicates = len(W) - 1,
			           loss_a = loss_a[0], loss_b = loss_b[0], p_a = (loss_a[1:] > loss_b[1:]).mean(),
			           mcda = INSTRUMENTS[int(np.argmax(W_NOMINAL * U[0, :, 0] + (1 - W_NOMINAL) * U[0, :, 1]))])
			for k, i in enumerate(INSTRUMENTS):
				row.update({f'forecast_{i}': U[0, k, 0], f'nowcast_{i}': U[0, k, 1],
				            f'acc_w_{i}': acc_w[k], f'acc_wc_{i}': acc_wc[k]})
			rows.append(row)
	return rows

def truth(raw):
	"""Per sigma_A: true difference of the losses (tie if not distinguishable from zero over the worlds) and true SMAA."""
	k = raw[(raw.engine == 'kalman') & (raw.n == max(LENGTHS))].assign(diff = lambda d: d.loss_a - d.loss_b)
	g = k.groupby('sigma_A')
	t = pd.DataFrame(dict(diff = g['diff'].mean(), se = g['diff'].std() / np.sqrt(g.size()),
	                      loss_a = g['loss_a'].mean(), loss_b = g['loss_b'].mean()))
	t['best'] = np.select([t['diff'] > 2 * t.se, t['diff'] < -2 * t.se], ['a', 'b'], 'tie')
	crit = g[[f'{c}_{i}' for i in INSTRUMENTS for c in ('forecast', 'nowcast')]].mean()
	for s, r in crit.iterrows():
		U = np.array([[r[f'forecast_{i}'], r[f'nowcast_{i}']] for i in INSTRUMENTS])
		acc = first_rank(U, WGRID)
		for k, i in enumerate(INSTRUMENTS):
			t.loc[s, f'acc_{i}'] = acc[k]
		t.loc[s, 'mcda'] = INSTRUMENTS[int(np.argmax(W_NOMINAL * U[:, 0] + (1 - W_NOMINAL) * U[:, 1]))]
	return t

def claim(values):
	"""Instrument that a SMAA claims as robustly best (first for at least DECIDE of the weights), or none."""
	for i, v in zip(INSTRUMENTS, values):
		if v >= DECIDE:
			return i
	return 'none'

def summarize(raw, t):
	"""Per engine, catalogue length and sigma_A: right answers, confident wrong answers and 'cannot tell' answers."""
	raw = raw.copy()
	raw['true_best'] = raw.sigma_A.map(t.best)
	raw['standard'] = np.where(raw.loss_a > raw.loss_b, 'a', 'b')
	raw['ours'] = np.select([raw.p_a >= DECIDE, raw.p_a <= 1 - DECIDE], ['a', 'b'], 'tie')
	# step 1c: a claim is wrong if the truth does not support it
	true_robust = t.apply(lambda r: claim([r[f'acc_{i}'] for i in INSTRUMENTS]), axis = 1)
	raw['true_robust'] = raw.sigma_A.map(true_robust)
	raw['true_mcda'] = raw.sigma_A.map(t.mcda)
	raw['claim_w'] = raw.apply(lambda r: claim([r[f'acc_w_{i}'] for i in INSTRUMENTS]), axis = 1)
	raw['claim_wc'] = raw.apply(lambda r: claim([r[f'acc_wc_{i}'] for i in INSTRUMENTS]), axis = 1)
	out = raw.assign(
		std_right = raw.standard == raw.true_best,
		ours_right = raw.ours == raw.true_best,
		ours_wrong = (raw.ours != raw.true_best) & (raw.ours != 'tie'),
		ours_cannot_tell = (raw.ours == 'tie') & (raw.true_best != 'tie'),
		mcda_wrong = raw.mcda != raw.true_mcda,
		w_wrong = (raw.claim_w != 'none') & (raw.claim_w != raw.true_robust),
		wc_wrong = (raw.claim_wc != 'none') & (raw.claim_wc != raw.true_robust),
		w_none = raw.claim_w == 'none', wc_none = raw.claim_wc == 'none')
	cols = ['std_right', 'ours_right', 'ours_wrong', 'ours_cannot_tell', 'mcda_wrong', 'w_wrong', 'wc_wrong', 'w_none', 'wc_none']
	s = out.groupby(['engine', 'n', 'sigma_A'])[cols].mean().reset_index()
	s['ours_answer'] = s.ours_right + s.ours_wrong				# share of worlds where our method names a winner
	s['ours_precision'] = s.ours_right / s.ours_answer.where(s.ours_answer > 0)	# right when it names one (the standard always does)
	return s

def crossing(t):
	"""Noise of A where the true losses of A and B are equal (linear interpolation)."""
	d = t['diff'].values
	k = np.flatnonzero(np.sign(d[:-1]) != np.sign(d[1:]))[0]
	x0, x1 = t.index[k], t.index[k + 1]
	return x0 + d[k] / (d[k] - d[k + 1]) * (x1 - x0)

ENGINES = dict(kalman = dict(color = '#235fa4', label = 'assimilative OSE'),		# blue: assimilative, red: statistical
               boosting = dict(color = '#b8322a', label = 'statistical OSE'))
LENGTH_LINES = {max(LENGTHS): dict(ls = '-', marker = 'o'), min(LENGTHS): dict(ls = '--', marker = 'o', mfc = 'white')}

def plot(t, summary):
	"""Top: step 1a. Bottom: step 1c. Left: the truth; middle and right: the methods against the noise of A."""
	fig, ax = plt.subplots(2, 3, figsize = (15, 8.5))
	x = t.index.values
	tie = crossing(t)
	ax[0, 0].plot(x, t.loss_a, 'k-^', label = 'A (unique contribution)')
	ax[0, 0].plot(x, t.loss_b, 'k--s', label = 'B (unique contribution)')
	for a in ax[0]:
		a.axvline(tie, ls = '-.', color = '0.5', lw = 0.9, label = f'no real winner (σ_A ≈ {tie:.1f})')
	ax[0, 0].set_ylabel('TSS loss when the instrument is removed')
	ax[0, 0].set_title('Step 1a truth')
	for i, style in zip(INSTRUMENTS[:2], ('-^', '--s')):
		ax[1, 0].plot(x, t[f'acc_{i}'], 'k' + style, label = f'{i.upper()} first')
	ax[1, 0].axhline(DECIDE, ls = ':', color = '0.4', lw = 0.9, label = f'robust winner (≥ {DECIDE:.0%})')
	ax[1, 0].set_ylabel('share of weights')
	ax[1, 0].set_title('Step 1c truth (SMAA on the true criteria)')
	for engine, es in ENGINES.items():
		for n, ls in LENGTH_LINES.items():
			s = summary[(summary.engine == engine) & (summary.n == n)]
			kw = dict(color = es['color'], **ls)
			ax[0, 1].plot(s.sigma_A, s.ours_answer, **kw)
			ax[0, 2].plot(s.sigma_A, s.std_right, alpha = 0.5, **dict(kw, ls = ':'))
			ax[0, 2].plot(s.sigma_A, s.ours_precision, **kw, label = f"{es['label']}, {n:,} steps")
			ax[1, 1].plot(s.sigma_A, s.mcda_wrong, alpha = 0.5, **dict(kw, ls = ':'))
			ax[1, 1].plot(s.sigma_A, s.w_wrong, **kw, label = f"{es['label']}, {n:,} steps")
			ax[1, 2].plot(s.sigma_A, s.w_wrong, alpha = 0.5, **dict(kw, ls = ':'))
			ax[1, 2].plot(s.sigma_A, s.wc_wrong, **kw)
	ax[0, 1].set_title('Step 1a: our method names a winner\n(the standard always does)')
	ax[0, 2].set_title('Step 1a: right when it names a winner\n(dotted: standard, solid: ours)')
	ax[1, 1].set_title('Step 1c: wrong claim\n(dotted: MCDA, solid: SMAA with weights only)')
	ax[1, 2].set_title('Step 1c: wrong claim of a robust winner\n(dotted: weights only, solid: weights + bootstrap)')
	for a in ax.flat:
		a.set_xlabel('noise of A (σ_A)')
	for a in list(ax[0, 1:]) + list(ax[1, 1:]):
		a.set_ylim(-0.03, 1.03)
		a.set_ylabel('fraction of worlds')
	for a in (ax[0, 0], ax[1, 0], ax[1, 1]):
		a.legend(loc = 'best', fontsize = 8)
	ax[0, 2].legend(loc = 'lower left', fontsize = 8)
	fig.tight_layout()
	fig.savefig(RES / 'run_1d.pdf')
	return fig

if __name__ == '__main__':
	if RAW.exists() and '--rerun' not in sys.argv:
		raw = pd.read_csv(RAW)
	else:
		rows = Parallel(n_jobs = -1, verbose = 5)(delayed(run_world)(seed, s) for seed in SEEDS for s in SIGMAS)
		raw = pd.DataFrame([r for world_rows in rows for r in world_rows])
		raw.to_csv(RAW, index = False)
	t = truth(raw)
	summary = summarize(raw, t)
	t.to_csv(RES / 'run_1d_truth.csv')
	summary.to_csv(RES / 'run_1d_summary.csv', index = False)
	print(t.round(3).to_string())
	print(summary.round(2).to_string(index = False))
	plot(t, summary)
	print(f'saved {RAW}, run_1d_truth.csv, run_1d_summary.csv, run_1d.pdf in {RES}')
