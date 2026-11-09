"""
Step 3: trial study. OSSE and design-space exploration for instrument A.
Every synthetic configuration of A (collecting area, integration time, saturation level) is simulated on a truth
and scored with the TSS of the forecast; then the best configuration within each budget and the Pareto front.
Truth: the observed world (ours: the catalogue) or nature runs from a model with a systematic error (standard OSSE).

python3 run_3.py            uses the saved OSSE results if they exist
python3 run_3.py --rerun    recomputes everything (about 3 minutes on 8 cores)
"""
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ose_assim import tss_max
from ose_stat import split, stat_forecast
from world import PARAMS, simulate

RES = Path(__file__).parent.parent/'results'
RAW = RES / 'run_3_osse.csv'
AREAS = tuple(np.round(np.geomspace(0.125, 4, 11), 3))	# collecting area, factor sqrt(2) apart (1 = instrument A of steps 1 and 2)
CADENCES = (1, 2, 3, 4, 6, 8)			# integration time [steps]
TOPS = (1, 1.5, 2, 3, 4.5, 6)			# saturation level
RANGE_COST = 0.05						# cost of the dynamic range, per unit of saturation level
RATE_COST = 0.1							# cost of the data rate, for one measurement per step
BUDGETS = dict(tight = 0.6, baseline = 1.2, wide = 2.5)
TRUTHS = dict(observed = PARAMS,							# the true world, i.e. the catalogue (ours)
              slow_source = dict(PARAMS, phi_d = 0.9),	# nature runs: the model gets the source too slow,
              wrong_lag = dict(PARAMS, L = 4),				# the lag too short,
              weak_source = dict(PARAMS, beta = 0.3))		# or the source effect too weak (standard OSSE)
SEED = 0								# same random numbers for every truth

def instrument(d, area, k, top, sigma_ref, rng):
	"""
	Synthetic A: averages the source over blocks of k steps, with noise sigma_ref / sqrt(area * k) and saturation at top;
	the value of a block is available from the step after the block ends
	"""
	T = len(d)
	block = np.arange(T) // k
	mean = np.bincount(block, d) / np.bincount(block)
	meas = np.minimum(mean + sigma_ref / np.sqrt(area * k) * rng.normal(size = len(mean)), top)
	out = np.zeros(T)
	out[k:] = np.repeat(meas, k)[:T - k]
	return out

def cost(df):
	"""
	Resource cost of each configuration (arbitrary units, like a mass): the area dominates,
	the range and the data rate (one value every k steps) cost little
	"""
	return df['area'] + RANGE_COST * df['top'] + RATE_COST / df['cadence']

def osse(params):
	"""
	TSS of every configuration of A, simulated on the world generated with params
	"""
	w = simulate(100_000, params, SEED)
	train, test = split(w)
	sigma_ref = params['sigma_A'] * w['d'].std()
	def score(cfg):
		a = instrument(w['d'], *cfg, sigma_ref, np.random.default_rng(1))
		prob = stat_forecast(dict(w, a = a), ('a',), train, 'boosting')
		return tss_max(w['y'][test], prob[test])
	grid = list(itertools.product(AREAS, CADENCES, TOPS))
	tss = Parallel(n_jobs = -1)(delayed(score)(cfg) for cfg in grid)
	df = pd.DataFrame(grid, columns = ['area', 'cadence', 'top']).assign(tss = tss)
	return df.assign(cost = cost(df))

def pareto(df):
	"""
	Configurations that no cheaper configuration beats
	"""
	df = df.sort_values(['cost', 'tss'], ascending = [True, False])
	return df[df['tss'] > df['tss'].cummax().shift(fill_value = -np.inf)]

def optima(results):
	"""
	For every truth and budget: the configuration chosen, the TSS expected from that truth and the TSS in the observed world
	"""
	observed = results['observed'].set_index(['area', 'cadence', 'top'])['tss']
	rows = []
	for truth, df in results.items():
		for budget, limit in BUDGETS.items():
			best = df[df['cost'] <= limit].sort_values('tss').iloc[-1]
			cfg = (best['area'], int(best['cadence']), best['top'])
			rows.append(dict(truth = truth, budget = budget, area = cfg[0], cadence = cfg[1], top = cfg[2],
			                 cost = best['cost'], tss_expected = best['tss'], tss_true = observed[cfg]))
	return pd.DataFrame(rows)

STYLES = dict(observed = dict(color = '#b8322a', marker = 'D', ls = '-', label = 'observed truth (ours)'),
              slow_source = dict(color = '#235fa4', marker = '^', ls = '--', label = 'nature run: source too slow'),
              wrong_lag = dict(color = '#0f8f80', marker = 's', ls = '--', label = 'nature run: lag too short'),
              weak_source = dict(color = '#8150b8', marker = 'o', ls = '--', label = 'nature run: source too weak'))
GREYS = dict(zip(CADENCES, np.linspace(0.3, 0.85, len(CADENCES)).astype(str)))	# cloud: darker = faster cadence

def plot(results, opt):
	"""
	Left: all configurations (observed truth) and Pareto front from each truth.
	Right: for each budget, TSS expected (open) and true (filled) of the chosen design
	"""
	fig, ax = plt.subplots(1, 2, figsize = (12, 4.8), gridspec_kw = dict(width_ratios = [1.3, 1]))
	obs = results['observed']
	clouds = [ax[0].scatter(g['cost'], g['tss'], s = 9, color = GREYS[k], lw = 0, label = f'{k}')
	          for k, g in obs.groupby('cadence')]
	ax[0].add_artist(ax[0].legend(handles = clouds, title = 'cadence [steps]\n(all configurations,\nobserved truth)',
	                              loc = 'upper left', fontsize = 7, title_fontsize = 7, markerscale = 1.5))
	fronts = [ax[0].plot(f['cost'], f['tss'], ms = 5, drawstyle = 'steps-post', **STYLES[truth])[0]
	          for truth, f in ((t, pareto(df)) for t, df in results.items())]
	for name, limit in BUDGETS.items():
		ax[0].axvline(limit, ls = '-.', color = '0.5', lw = 0.9)
		ax[0].text(limit, 0.01, f' {name}', transform = ax[0].get_xaxis_transform(), va = 'bottom', fontsize = 8, color = '0.3')
	ax[0].set_xscale('log')
	ticks = [0.2, 0.3, 0.5, 1, 2, 4]
	ax[0].set_xticks(ticks, [f'{t:g}' for t in ticks])
	ax[0].minorticks_off()
	ax[0].set_xlabel('cost (area + range + data rate)')
	ax[0].set_ylabel('TSS of the forecast')
	ax[0].set_title(f'Pareto front of A, as seen from each truth ({len(obs)} configurations)')
	ax[0].legend(handles = fronts, loc = 'lower right', fontsize = 8)
	offsets = np.linspace(-0.27, 0.27, len(TRUTHS))
	for (truth, style), dx in zip(STYLES.items(), offsets):
		o = opt[opt['truth'] == truth]
		x = np.arange(len(BUDGETS)) + dx
		ax[1].vlines(x, o['tss_true'], o['tss_expected'], color = style['color'], lw = 1.2)
		ax[1].plot(x, o['tss_expected'], ls = 'none', marker = style['marker'], mfc = 'white', mec = style['color'], ms = 7)
		ax[1].plot(x, o['tss_true'], ls = 'none', marker = style['marker'], color = style['color'], ms = 7)
	ax[1].plot([], [], ls = 'none', marker = 'o', mfc = 'white', mec = '0.3', label = 'expected from that truth')
	ax[1].plot([], [], ls = 'none', marker = 'o', color = '0.3', label = 'true (in the observed world)')
	ax[1].set_xticks(range(len(BUDGETS)), list(BUDGETS))
	ax[1].set_xlabel('budget')
	ax[1].set_title('Chosen configuration: expected and true TSS')
	ax[1].legend(loc = 'lower right', fontsize = 8)
	fig.tight_layout()
	fig.savefig(RES / 'run_3.pdf')
	return fig

if __name__ == '__main__':
	if RAW.exists() and '--rerun' not in sys.argv:
		raw = pd.read_csv(RAW)
		results = {truth: g.drop(columns = 'truth').reset_index(drop = True) for truth, g in raw.groupby('truth', sort = False)}
	else:
		results = {truth: osse(params) for truth, params in TRUTHS.items()}
		pd.concat(results, names = ['truth']).reset_index(level = 0).to_csv(RAW, index = False)
	opt = optima(results)
	opt.to_csv(RES / 'run_3_optima.csv', index = False)
	for truth, df in results.items():
		print(f'{truth}: Pareto front {len(pareto(df))} of {len(df)} configurations')
	print(opt.round(3).to_string(index = False))
	plot(results, opt)
	print(f'\nsaved {RAW}, {RES / "run_3_optima.csv"}, {RES / "run_3.pdf"}')
