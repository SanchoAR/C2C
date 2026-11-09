"""
Step 1a trial study: when do the assimilative and the statistical OSE recover the true instrument ranking?
Grid: model error (Kalman filter) x catalogue length (statistical engines), repeated over several synthetic worlds.
Truth: Kalman filter with the perfect model, on the same world and the same test steps.

python3 run_1a.py            uses the saved raw results if they exist
python3 run_1a.py --rerun    recomputes everything (about 5 minutes on 8 cores)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ose import denial
from ose_assim import assim_forecast
from ose_stat import split, stat_forecast
from world import PARAMS, simulate

RES = Path(__file__).parent.parent / 'results'
RAW = RES / 'run_1a_raw.csv'
SUMMARY = RES / 'run_1a_summary.csv'

SEEDS = range(10)											# one synthetic world per seed
MODELS = {'true': PARAMS,									# models given to the Kalman filter
          'L = 5': dict(PARAMS, L = 5), 'L = 4': dict(PARAMS, L = 4), 'L = 3': dict(PARAMS, L = 3),
          'beta = 0.4': dict(PARAMS, beta = 0.4), 'beta = 0.3': dict(PARAMS, beta = 0.3),
          'beta = 0.2': dict(PARAMS, beta = 0.2)}
LAG_SERIES = ['true', 'L = 5', 'L = 4', 'L = 3']			# model error grows along each series
BETA_SERIES = ['true', 'beta = 0.4', 'beta = 0.3', 'beta = 0.2']
LENGTHS = [50_000, 20_000, 5_000, 2_000, 1_000, 500]		# training steps for the statistical engines
LEARNERS = ('logistic', 'boosting')
RELIABLE, UNCERTAIN = 0.95, 0.80							# thresholds on the fraction of correct choices


def losses(tss):
	"""Leave-one-out losses (TSS with all instruments minus TSS without each one)."""
	return dict(tss_all = tss['all'], **{f'loss_{i}': tss['all'] - tss[f'without {i}'] for i in 'abc'})


def run_seed(seed):
	"""All grid cells for one synthetic world: one row per engine and setting."""
	w = simulate(100_000, PARAMS, seed)
	train, test = split(w)
	rows = []
	for name, model in MODELS.items():
		engine = lambda w, inst, tr, m = model: assim_forecast(w, inst, tr, m)
		tss = denial(engine, w, train, test, alone = False)
		rows.append(dict(seed = seed, engine = 'kalman', setting = name, **losses(tss)))
	idx = np.flatnonzero(train)
	for n in LENGTHS:
		short = np.zeros_like(train)
		short[idx[-n:]] = True								# the n training steps just before the test period
		for learner in LEARNERS:
			engine = lambda w, inst, tr, l = learner: stat_forecast(w, inst, tr, l)
			tss = denial(engine, w, short, test, alone = False)
			rows.append(dict(seed = seed, engine = learner, setting = str(n), **losses(tss)))
	return rows


def summarize(raw):
	"""Per grid cell: fraction of worlds where the best instrument is the true one, and share of A's value seen."""
	cols = ['loss_a', 'loss_b', 'loss_c']
	truth = raw[(raw.engine == 'kalman') & (raw.setting == 'true')].set_index('seed')
	raw = raw.assign(best = raw[cols].idxmax(axis = 1).str[-1],
	                 true_best = raw.seed.map(truth[cols].idxmax(axis = 1).str[-1]),
	                 value_a = raw.loss_a / raw.seed.map(truth.loss_a))
	raw['correct'] = raw.best == raw.true_best
	summary = raw.groupby(['engine', 'setting'], sort = False).agg(
		correct = ('correct', 'mean'), value_a = ('value_a', 'median'),
		value_a_lo = ('value_a', lambda v: v.quantile(0.25)), value_a_hi = ('value_a', lambda v: v.quantile(0.75)),
		tss_all = ('tss_all', 'mean'), n_worlds = ('seed', 'size')).reset_index()
	summary['verdict'] = np.select([summary.correct >= RELIABLE, summary.correct >= UNCERTAIN],
	                               ['reliable', 'uncertain'], 'not reliable')
	return summary


STYLES = dict(													# blue for the assimilative OSE, warm colours for the statistical one
	lag = dict(color = '#235fa4', marker = '^', label = 'lag error (L = 6 → 3)'),
	beta = dict(color = '#4a95d6', marker = '+', mew = 2, label = 'amplitude error (β = 0.5 → 0.2)'),
	logistic = dict(color = '#b8322a', marker = 'D', label = 'logistic'),
	boosting = dict(color = '#d97a10', marker = 'x', mew = 2, label = 'boosting'),
)


def plot(summary):
	"""Top: fraction of correct choices; bottom: share of A's value seen. Left: model error; right: catalogue length."""
	fig, ax = plt.subplots(2, 2, figsize = (10, 7), sharey = 'row')
	k = summary[summary.engine == 'kalman'].set_index('setting')
	for key, series in [('lag', LAG_SERIES), ('beta', BETA_SERIES)]:
		x = np.arange(len(series))
		ax[0, 0].plot(x, k.loc[series, 'correct'], ms = 7, **STYLES[key])
		ax[1, 0].plot(x, k.loc[series, 'value_a'], ms = 7, **STYLES[key])
		ax[1, 0].fill_between(x, k.loc[series, 'value_a_lo'], k.loc[series, 'value_a_hi'],
		                      color = STYLES[key]['color'], alpha = 0.2)
	for a in ax[:, 0]:
		a.set_xticks(range(4), ['none', 'small', 'medium', 'large'])
	ax[1, 0].set_xlabel('model error of the Kalman filter')
	ax[0, 0].set_title('Assimilative OSE')
	for learner in LEARNERS:
		s = summary[summary.engine == learner]
		n = s.setting.astype(int)
		ax[0, 1].plot(n, s.correct, ms = 7, **STYLES[learner])
		ax[1, 1].plot(n, s.value_a, ms = 7, **STYLES[learner])
		ax[1, 1].fill_between(n, s.value_a_lo, s.value_a_hi, color = STYLES[learner]['color'], alpha = 0.2)
	for a in ax[:, 1]:
		a.set_xscale('log')
		a.set_xticks(LENGTHS, [f'{n:,}' for n in LENGTHS])
		a.minorticks_off()
		a.invert_xaxis()									# shorter catalogue to the right, like larger model error
	ax[1, 1].set_xlabel('training steps (catalogue length)')
	ax[0, 1].set_title('Statistical OSE')
	for a in ax[0]:
		a.axhline(RELIABLE, ls = '--', color = '0.3', lw = 0.9, label = f'reliable (≥ {RELIABLE:.0%})')
		a.axhline(UNCERTAIN, ls = ':', color = '0.3', lw = 0.9, label = f'uncertain ({UNCERTAIN:.0%}–{RELIABLE:.0%})')
		a.legend(loc = 'lower left')
	for a in ax[1]:
		a.axhline(1, ls = '-.', color = '0.3', lw = 0.9, label = "true value of A")
		a.legend(loc = 'best')
	ax[0, 0].set_ylabel('fraction of worlds where\nthe best instrument is right')
	ax[1, 0].set_ylabel("share of A's true value seen\n(median, 25–75%)")
	fig.tight_layout()
	fig.savefig(RES / 'run_1a.pdf')
	return fig


if __name__ == '__main__':
	if RAW.exists() and '--rerun' not in sys.argv:
		raw = pd.read_csv(RAW, dtype = {'setting': str})
	else:
		rows = Parallel(n_jobs = -1, verbose = 5)(delayed(run_seed)(s) for s in SEEDS)
		raw = pd.DataFrame([r for seed_rows in rows for r in seed_rows])
		raw.to_csv(RAW, index = False)
	summary = summarize(raw)
	summary.to_csv(SUMMARY, index = False)
	print(summary.round(3).to_string(index = False))
	plot(summary)
	print(f'saved {RAW}, {SUMMARY}, {RES / "run_1a.pdf"}')
