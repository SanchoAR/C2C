"""
Step 4: trial study. Robustness of the results to the finite length of the catalogue.
The synthetic world gives what reality does not: many independent catalogues, hence the true spread of the results.
Both engines: statistical OSE (boosting, trained on the catalogue) and assimilative OSE (Kalman filter with
the perfect model, no training), fed with the same synthetic configurations of A.
A) Uncertainty of the TSS of the three step-3 optima: bootstrap over independent steps (standard) and over blocks
   of steps (ours), checked against the spread across worlds.
B) Robustness of the step-3 choice: the optimum of a single catalogue (standard) and the probability of being
   the optimum from the paired block bootstrap, with the set of equivalent configurations (ours).
Needs the results of run_3.py (configurations near the Pareto front of the observed truth).

python3 run_4.py            uses the saved raw results if they exist
python3 run_4.py --rerun    recomputes everything (about 20 minutes on 8 cores)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ose_assim import assim_forecast_instrument
from ose_stat import split, stat_forecast
from run_3 import BUDGETS, cost, instrument
from world import PARAMS, simulate

RES = Path(__file__).parent.parent/'results'
RAW_A = RES / 'run_4_spread_raw.csv'
RAW_B = RES / 'run_4_choice_raw.csv'
SEEDS = range(100, 130)					# independent worlds (seed 0, used in step 3, is left out)
LENGTHS = [50_000, 20_000, 10_000, 5_000, 2_000]	# test steps of the catalogue (2,000 steps: about 18 episodes)
BLOCKS = {'independent steps': 1, 'blocks of 10': 10, 'blocks of 50': 50, 'blocks of 200': 200}
DECISION = 'blocks of 50'				# bootstrap of our method (part B and the main lines of part A)
N_BOOT = 200							# bootstrap replicates
NEAR = 0.03								# part B: configurations within this TSS of the optimum, for each budget
COVER = 0.90							# nominal level of the intervals and of the set of equivalent configurations
CFG = ['area', 'cadence', 'top']

def candidates():
	"""
	From step 3 (observed truth): the configurations near the optimum of each budget (part B)
	and the three optima (part A)
	"""
	osse = pd.read_csv(RES / 'run_3_osse.csv')
	obs = osse[osse['truth'] == 'observed']
	near = pd.concat([obs[(obs['cost'] <= limit) & (obs['tss'] >= obs.loc[obs['cost'] <= limit, 'tss'].max() - NEAR)]
	                  for limit in BUDGETS.values()]).drop_duplicates(CFG)
	opt = pd.read_csv(RES / 'run_3_optima.csv')
	optima = opt[opt['truth'] == 'observed']
	as_cfg = lambda df: [(a, int(k), t) for a, k, t in df[CFG].values]	# the cadence is a number of steps
	return as_cfg(near), as_cfg(optima)

def weights(n, block, rng):
	"""
	Circular block bootstrap: how many times each of the n steps is drawn, in each replicate (block = 1: independent steps)
	"""
	nb = -(-n // block)
	starts = rng.integers(0, n, size = (N_BOOT, nb))
	idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(N_BOOT, -1)[:, :n]
	return np.stack([np.bincount(row, minlength = n) for row in idx]).astype(np.float32)

def tss_weighted(y, prob, W):
	"""
	Best TSS over all probability thresholds, with step weights W (one row per replicate);
	with all weights equal to one it is tss_max of ose_assim
	"""
	W = np.atleast_2d(W)
	order = np.argsort(-prob, kind = 'stable')
	p, pos, Wo = prob[order], y[order].astype(bool), W[:, order]
	last = np.r_[p[1:] != p[:-1], True]								# thresholds between distinct probabilities
	tp = np.cumsum(Wo * pos, axis = 1)[:, last]
	fp = np.cumsum(Wo * ~pos, axis = 1)[:, last]
	return np.max(tp / tp[:, -1:] - fp / fp[:, -1:], axis = 1)

def run_world(seed, near, optima):
	"""
	One independent world: forecasts of every case on the test steps, then for every catalogue length
	the TSS, its bootstrap spread (part A, both engines) and the probability of each configuration being the optimum (part B)
	"""
	w = simulate(100_000, PARAMS, seed)
	train, test = split(w)							# long training period, the same for every catalogue length
	idx = np.flatnonzero(test)
	sigma_ref = PARAMS['sigma_A'] * w['d'].std()
	probs = dict(kalman = {}, boosting = {})
	for cfg in near:
		a = instrument(w['d'], *cfg, sigma_ref, np.random.default_rng(seed))
		probs['kalman'][cfg] = assim_forecast_instrument(w, a, *cfg, sigma_ref, where = test)[idx]
		probs['boosting'][cfg] = stat_forecast(dict(w, a = a), ('a',), train, 'boosting')[idx]
	cases = {(engine, f'A {a:g}/{k}/{t:g}'): probs[engine][(a, k, t)] for engine in probs for a, k, t in optima}
	y, event = w['y'][idx], w['event'][idx]
	rng = np.random.default_rng(seed)
	rows_a, rows_b = [], []
	for n in LENGTHS:
		yn = y[:n]
		m = len(yn)									# valid test steps (the longest catalogue loses the last few)
		episodes = int((np.diff(event[:n]) == 1).sum())
		point = {engine: {cfg: tss_weighted(yn, pe[cfg][:n], np.ones(m))[0] for cfg in near} for engine, pe in probs.items()}
		point_cases = {(engine, case): point[engine][cfg] for engine, case, cfg in
		               ((e, f'A {a:g}/{k}/{t:g}', (a, k, t)) for e in probs for a, k, t in optima)}
		for method, block in BLOCKS.items():
			W = weights(m, block, rng)				# the same replicates for every case (paired bootstrap)
			W = W[(W @ yn > 0) & (W @ (1 - yn) > 0)]	# drop the rare replicates without events (no TSS)
			for (engine, case), prob in cases.items():
				boot = tss_weighted(yn, prob[:n], W)
				lo, hi = np.quantile(boot, [(1 - COVER) / 2, (1 + COVER) / 2])
				rows_a.append(dict(seed = seed, n = n, episodes = episodes, method = method, engine = engine, case = case,
				                   tss = point_cases[(engine, case)], sd = boot.std(), lo = lo, hi = hi))
			if method != DECISION:
				continue
			for engine, pe in probs.items():
				boot = {cfg: tss_weighted(yn, pe[cfg][:n], W) for cfg in near}
				for budget, limit in BUDGETS.items():
					within = [cfg for cfg in near if cost(dict(zip(CFG, cfg))) <= limit]
					wins = np.argmax(np.column_stack([boot[cfg] for cfg in within]), axis = 1)
					p_best = np.bincount(wins, minlength = len(within)) / len(W)
					for cfg, p in zip(within, p_best):
						rows_b.append(dict(seed = seed, n = n, engine = engine, budget = budget, **dict(zip(CFG, cfg)),
						                   tss = point[engine][cfg], p_best = p))
	return rows_a, rows_b

def summarize_spread(raw):
	"""
	Part A: bootstrap standard deviation over the true one (spread across worlds) and coverage of the intervals,
	pooled over the cases of each engine
	"""
	keys = ['engine', 'case', 'n']
	truth = raw[raw['method'] == DECISION].groupby(keys)['tss'].agg(true_sd = 'std', target = 'mean')
	raw = raw.join(truth, on = keys)
	raw['ratio'] = raw['sd'] / raw['true_sd']
	raw['covered'] = (raw['lo'] <= raw['target']) & (raw['target'] <= raw['hi'])
	return raw.groupby(['engine', 'method', 'n'], sort = False).agg(
		episodes = ('episodes', 'mean'), ratio = ('ratio', 'median'),
		ratio_lo = ('ratio', lambda v: v.quantile(0.25)), ratio_hi = ('ratio', lambda v: v.quantile(0.75)),
		coverage = ('covered', 'mean')).reset_index()

def summarize_choice(raw):
	"""
	Part B, for every engine, catalogue length and budget. True optimum: best mean TSS across worlds on the longest
	catalogue. Standard: the single-catalogue optimum. Ours: the fewest configurations that reach COVER of the
	bootstrap probability.
	"""
	ref = raw[raw['n'] == max(LENGTHS)].groupby(['engine', 'budget'] + CFG)['tss'].mean()
	rows = []
	for (engine, n, budget), g in raw.groupby(['engine', 'n', 'budget'], sort = False):
		pop = ref.loc[(engine, budget)]
		best = pop.idxmax()
		for seed, s in g.groupby('seed'):
			s = s.set_index(CFG)
			chosen = s['tss'].idxmax()
			ranked = s['p_best'].sort_values(ascending = False)
			eq = ranked.index[:np.searchsorted(ranked.cumsum().values, COVER - 1e-9) + 1]
			rows.append(dict(engine = engine, n = n, budget = budget, std_right = chosen == best,
			                 std_loss = pop.max() - pop[chosen], p_boot = s.loc[best, 'p_best'], set_has_best = best in eq,
			                 set_size = len(eq), set_worst_loss = pop.max() - pop[eq].min()))
	return pd.DataFrame(rows).groupby(['engine', 'n', 'budget'], sort = False).agg(
		std_right = ('std_right', 'mean'), std_loss_mean = ('std_loss', 'mean'), std_loss_max = ('std_loss', 'max'),
		p_boot = ('p_boot', 'mean'), set_has_best = ('set_has_best', 'mean'), set_size = ('set_size', 'mean'),
		set_worst_loss = ('set_worst_loss', 'mean')).reset_index()

ENGINES = dict(kalman = dict(color = '#235fa4', marker = 'o', label = 'assimilative OSE'),	# blue: assimilative
               boosting = dict(color = '#b8322a', marker = 'D', label = 'statistical OSE'))	# warm: statistical
BOOT_LINES = {'independent steps': dict(ls = '--', mfc = 'white', label = 'independent steps (standard)'),
              DECISION: dict(ls = '-', label = f'{DECISION} steps (ours)')}
LENGTH_LINES = {max(LENGTHS): dict(ls = '-'), min(LENGTHS): dict(ls = '--', mfc = 'white')}	# block-length panel:
																							# longest and shortest catalogue

def plot(spread, choice):
	"""
	Top (part A, both engines): bootstrap spread over the true spread, coverage of the 90% intervals, block length.
	Bottom (part B): does the step-3 choice find the true optimum? One panel per engine, then the size of the set
	"""
	fig, ax = plt.subplots(2, 3, figsize = (16, 9))
	for engine, es in ENGINES.items():
		for method, bs in BOOT_LINES.items():
			s = spread[(spread['engine'] == engine) & (spread['method'] == method)]
			style = dict(color = es['color'], marker = es['marker'], ms = 6, ls = bs['ls'], mfc = bs.get('mfc', es['color']),
			             label = f'{es["label"]}, {bs["label"]}')
			ax[0, 0].plot(s['n'], s['ratio'], **style)
			ax[0, 0].fill_between(s['n'], s['ratio_lo'], s['ratio_hi'], color = es['color'], alpha = 0.12)
			ax[0, 1].plot(s['n'], s['coverage'], **style)
		for n, ln in LENGTH_LINES.items():
			s = spread[(spread['engine'] == engine) & (spread['n'] == n)].set_index('method').loc[list(BLOCKS)]
			ax[0, 2].plot(list(BLOCKS.values()), s['ratio'], color = es['color'], marker = es['marker'], ms = 6, **ln,
			              label = f'{es["label"]}, {n:,} steps (~{s["episodes"].iloc[0]:.0f} episodes)')
	ax[0, 0].axhline(1, ls = '-.', color = '0.3', lw = 0.9, label = 'true spread')
	ax[0, 0].set_ylim(0, 1.3)						# room for the legend below the curves
	ax[0, 0].set_ylabel('bootstrap SD of the TSS / true SD\n(median, 25–75% over worlds and cases)')
	ax[0, 0].set_title('A. Is the uncertainty right?')
	ax[0, 1].axhline(COVER, ls = '-.', color = '0.3', lw = 0.9, label = f'nominal {COVER:.0%}')
	ax[0, 1].set_ylim(0, 1.02)
	ax[0, 1].set_ylabel(f'coverage of the {COVER:.0%} intervals')
	ax[0, 1].set_title('A. Do the intervals hold the true value?')
	ax[0, 2].axhline(1, ls = '-.', color = '0.3', lw = 0.9, label = 'true spread')
	ax[0, 2].set_ylim(0, 1.05)
	ax[0, 2].set_xscale('log')
	ax[0, 2].set_xticks(list(BLOCKS.values()), ['1\n(independent)', '10', '50', '200'])
	ax[0, 2].minorticks_off()
	ax[0, 2].set_xlabel('block length [steps]')
	ax[0, 2].set_ylabel('bootstrap SD of the TSS / true SD (median)')
	ax[0, 2].set_title('A. How long must the blocks be?')
	for a, (engine, es) in zip(ax[1, :2], ENGINES.items()):
		c = choice[choice['engine'] == engine].groupby('n', sort = False).mean(numeric_only = True).reset_index()
		a.plot(c['n'], c['std_right'], color = es['color'], ls = '--', marker = es['marker'], mfc = 'white', ms = 6,
		       label = 'single-catalogue optimum is the true one (standard)')
		a.plot(c['n'], c['p_boot'], color = es['color'], ls = ':', lw = 1.2,
		       label = 'bootstrap probability of the true optimum (ours)')
		a.plot(c['n'], c['set_has_best'], color = es['color'], ls = '-', marker = es['marker'], ms = 6,
		       label = f'set of equivalent configurations ({COVER:.0%})\nholds the true optimum (ours)')
		a.axhline(COVER, ls = '-.', color = '0.3', lw = 0.9, label = f'nominal {COVER:.0%}')
		a.set_ylim(0, 1.05)
		a.set_ylabel('fraction of worlds')
		a.set_title(f'B. Is the step-3 choice the true optimum? ({es["label"]})', fontsize = 10)
		ax[1, 2].plot(c['n'], c['set_size'], color = es['color'], marker = es['marker'], ms = 6, label = es['label'])
	ax[1, 2].set_ylim(0, None)
	ax[1, 2].set_ylabel('configurations in the set (mean over budgets and worlds)')
	ax[1, 2].set_title('B. How many configurations are equivalent?')
	episodes = spread[spread['method'] == DECISION].groupby('n')['episodes'].mean()
	for a in (ax[0, 0], ax[0, 1], *ax[1]):
		a.set_xscale('log')
		a.set_xticks(LENGTHS, [f'{n:,}\n(~{episodes[n]:.0f} ep.)' for n in LENGTHS])
		a.minorticks_off()
		a.invert_xaxis()								# shorter catalogue to the right, as in run_1a
		a.set_xlabel('test steps of the catalogue (episodes)')
	for a in ax.flat:
		a.legend(loc = 'upper left' if a is ax[1, 2] else 'lower right' if a is ax[0, 2] else 'lower left', fontsize = 7)
	fig.tight_layout()
	fig.savefig(RES / 'run_4.pdf')
	return fig

if __name__ == '__main__':
	if RAW_A.exists() and RAW_B.exists() and '--rerun' not in sys.argv:
		raw_a, raw_b = pd.read_csv(RAW_A), pd.read_csv(RAW_B)
	else:
		near, optima = candidates()
		print(f'{len(near)} configurations near the optimum, {len(SEEDS)} worlds')
		out = Parallel(n_jobs = -1, verbose = 5)(delayed(run_world)(s, near, optima) for s in SEEDS)
		raw_a = pd.DataFrame([r for rows, _ in out for r in rows])
		raw_b = pd.DataFrame([r for _, rows in out for r in rows])
		raw_a.to_csv(RAW_A, index = False)
		raw_b.to_csv(RAW_B, index = False)
	spread, choice = summarize_spread(raw_a), summarize_choice(raw_b)
	spread.to_csv(RES / 'run_4_spread.csv', index = False)
	choice.to_csv(RES / 'run_4_choice.csv', index = False)
	print('A. bootstrap SD / true SD (median over worlds and cases) and coverage of the 90% intervals')
	print(spread.round(3).to_string(index = False))
	print('\nB. choice of the configuration (fractions over worlds; losses in TSS against the true optimum)')
	print(choice.round(3).to_string(index = False))
	plot(spread, choice)
	print(f'\nsaved {RAW_A}, {RAW_B}, {RES / "run_4_spread.csv"}, {RES / "run_4_choice.csv"}, {RES / "run_4.pdf"}')
