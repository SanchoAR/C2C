"""
Step 1c trial study: how much does the instrument ranking depend on the weights of the criteria?
Criteria (each instrument alone): forecast TSS (event in the next H steps) and nowcast TSS (event now).
Standard MCDA: one weight vector, one winner. SMAA: many random weight vectors, frequency of each rank.

python3 run_1c.py    (about 30 seconds)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ose_assim import assim_forecast, assim_nowcast, tss_max
from ose_stat import split, stat_forecast
from world import PARAMS, simulate

RES = Path(__file__).parent.parent / 'results'

INSTRUMENTS = ('a', 'b', 'c')
CRITERIA = dict(forecast = 'y', nowcast = 'event')			# criterion: label it is verified against
ENGINES = dict(												# baseline engines: perfect model, long catalogue
	kalman = dict(forecast = assim_forecast, nowcast = assim_nowcast),
	boosting = dict(forecast = lambda w, inst, tr: stat_forecast(w, inst, tr, 'boosting', target = 'y'),
	                nowcast = lambda w, inst, tr: stat_forecast(w, inst, tr, 'boosting', target = 'event')),
)
TITLES = dict(kalman = 'Assimilative OSE (Kalman filter)', boosting = 'Statistical OSE (gradient boosting)')
W_NOMINAL = 0.5												# weight of the forecast criterion in the standard MCDA
N_WEIGHTS = 20_000											# random weight vectors in the SMAA

SHADES = dict(kalman = ['#1f4e8c', '#5b8fd0', '#bcd5f0'],		# blue for the assimilative OSE (1st, 2nd, 3rd rank)
              boosting = ['#9e2a1f', '#e0702a', '#f7c99b'])		# warm colours for the statistical OSE
LINES = dict(a = dict(ls = '-', marker = '^'), b = dict(ls = '--', marker = 's'), c = dict(ls = ':', marker = 'o'))


def criteria_matrix(engine, w, train, test):
	"""TSS on the test steps of each instrument alone (rows) for each criterion (columns)."""
	return pd.DataFrame({crit: [tss_max(w[label][test], engine[crit](w, (i,), train)[test]) for i in INSTRUMENTS]
	                     for crit, label in CRITERIA.items()}, index = INSTRUMENTS)


def smaa(U, n = N_WEIGHTS, seed = 0):
	"""Rank acceptability: fraction of random weight vectors (uniform Dirichlet) that give each instrument each rank."""
	W = np.random.default_rng(seed).dirichlet(np.ones(U.shape[1]), n)
	rank = np.argsort(np.argsort(-(W @ U.T), axis = 1), axis = 1)		# 0 = first place
	return pd.DataFrame([[(rank[:, i] == r).mean() for r in range(len(U))] for i in range(len(U))],
	                    index = INSTRUMENTS, columns = ['1st', '2nd', '3rd'])


def plot(mats, accs):
	"""Left: weighted score against the forecast weight. Right: rank acceptability. One row per engine."""
	fig, ax = plt.subplots(2, 2, figsize = (11, 7), gridspec_kw = dict(width_ratios = [1.4, 1]))
	wgrid = np.linspace(0, 1, 1001)
	for row, name in enumerate(ENGINES):
		U, acc, shades = mats[name], accs[name], SHADES[name]
		scores = np.outer(wgrid, U['forecast']) + np.outer(1 - wgrid, U['nowcast'])
		for i, inst in enumerate(INSTRUMENTS):
			ax[row, 0].plot(wgrid, scores[:, i], color = shades[0], ms = 7, markevery = 100, label = inst.upper(), **LINES[inst])
		winner = scores.argmax(axis = 1)
		for k, x in enumerate(wgrid[1:][winner[1:] != winner[:-1]]):
			ax[row, 0].axvline(x, ls = ':', color = '0.3', lw = 0.9, label = None if k else f'winner changes (w = {x:.2f})')
		ax[row, 0].axvline(W_NOMINAL, ls = '--', color = '0.3', lw = 0.9, label = f'single weight of the standard MCDA (w = {W_NOMINAL})')
		ax[row, 0].set_ylim(0.15, 0.95)						# room for the legend below the curves
		ax[row, 0].set_ylabel('weighted score (TSS)')
		ax[row, 0].set_title(TITLES[name])
		ax[row, 0].legend(loc = 'lower left', fontsize = 8)
		left = np.zeros(len(acc))
		for r, col in enumerate(acc.columns):
			ax[row, 1].barh(acc.index.str.upper(), acc[col], left = left, color = shades[r], edgecolor = 'white', label = f'{col} place')
			for y, (l, v) in enumerate(zip(left, acc[col])):
				if v >= 0.08:
					ax[row, 1].text(l + v / 2, y, f'{v:.0%}', ha = 'center', va = 'center', fontsize = 9,
					                color = 'white' if r < 2 else 'black')
			left += acc[col].values
		ax[row, 1].invert_yaxis()
		ax[row, 1].set_xlim(0, 1)
		ax[row, 1].set_title('SMAA: rank acceptability')
		ax[row, 1].legend(loc = 'lower right', fontsize = 8)
	ax[1, 0].set_xlabel('weight of the forecast criterion w   (nowcast weight = 1 − w)')
	ax[1, 1].set_xlabel('share of random weight vectors')
	fig.tight_layout()
	fig.savefig(RES / 'run_1c.pdf')
	return fig


if __name__ == '__main__':
	w = simulate(100_000, PARAMS)
	train, test = split(w)
	mats = {name: criteria_matrix(engine, w, train, test) for name, engine in ENGINES.items()}
	accs = {name: smaa(U.values) for name, U in mats.items()}
	for name in ENGINES:
		U = mats[name]
		nominal = W_NOMINAL * U['forecast'] + (1 - W_NOMINAL) * U['nowcast']
		print(f'\n{name}: criteria (TSS of each instrument alone)\n{U.round(3)}')
		print(f'standard MCDA (w = {W_NOMINAL}): {nominal.round(3).to_dict()} -> winner {nominal.idxmax().upper()}')
		print(f'SMAA rank acceptability\n{accs[name].round(3)}')
	pd.concat(mats, names = ['engine', 'instrument']).to_csv(RES / 'run_1c_criteria.csv')
	pd.concat(accs, names = ['engine', 'instrument']).to_csv(RES / 'run_1c_smaa.csv')
	plot(mats, accs)
	print(f'\nsaved {RES / "run_1c_criteria.csv"}, {RES / "run_1c_smaa.csv"}, {RES / "run_1c.pdf"}')
