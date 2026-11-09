"""
Step 2: trial study. Requirements of instrument A (cadence, latency, dynamic range)
from climatological statistics (standard) and from event-conditioned catalogue statistics (ours),
checked against the true requirement (the value that keeps 95% of the full TSS)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ose_assim import tss_max
from ose_stat import split, stat_forecast
from world import PARAMS, simulate

RES = Path(__file__).parent.parent/'results'
KEEP = 0.95								# a requirement is met if A keeps this share of its full TSS
COVER = 0.90							# share of event episodes the catalogue requirements must cover
WINDOW = 30								# steps before an event onset searched for the precursor peak
CADENCES = range(1, 9)					# values tried for the true requirement
LATENCIES = range (0, 9)
TOPS = np.arange(6.0, 0.4, -0.5)		# saturation levels, from loose to strict

def with_cadence(a, k):
	"""
	A measures every k steps and holds the last value in between
	"""
	idx = np.arange(len(a)) // k * k
	return a[idx]

def with_latency(a, k):
	"""
	The measurement of step t is available only at step t + k (zero before the first one arrives)
	"""
	out = np.zeros_like(a)
	out[k:] = a[:len(a) - k]
	return out

def with_saturation(a, top):
	"""
	A cannot measure above top
	"""
	return np.minimum(a, top)

def tss_of(w, a2, train, test):
	"""
	TSS of instrument A alone when its measurements are replaced by a2 (statistical engine, boosting)
	"""
	prob = stat_forecast(dict(w, a = a2), ('a',), train, 'boosting')
	return tss_max(w['y'][test], prob[test])

def climatology(a):
	"""
	Standard characterisation, from the whole record: correlation time and extreme values of A
	"""
	T = len(a)

	acf = [np.corrcoef(a[:T-k], a[k:])[0, 1] for k in range(1, WINDOW)]
	tau = next(k for k, r in enumerate(acf, start = 1) if r < 1 / np.e)		# e-folding time of the autocorrelation
	return dict(correlation_time = tau, q999 = np.quantile(a, 0.999))

def episodes(a, event):
	"""
	Event-conditioned characterisation, from the catalogue: for every episode onset,
	how long before it the precursor peaks (lead), how long the precursor takes to rise, and how high it gets
	"""
	onsets = np.flatnonzero(np.diff(event) == 1) + 1
	onsets = [o for o in onsets if o > WINDOW and event[o - 10:o].sum() == 0]	# first onset of each episode
	median = np.median(a)
	rows = []
	for o in onsets:
		win = a[o - WINDOW:o]
		peak = np.argmax(win)
		below = np.flatnonzero(win[:peak] < median)
		rows.append(dict(lead = WINDOW-peak, rise = peak - below[-1] if len(below) else np.nan, peak_value = win[peak]))
	return pd.DataFrame(rows)

def requirements(clim, ep):
	"""
	Requirements derived from the statistics, before any forecast is run
	"""
	clim_req = dict(cadence = max(1, clim['correlation_time'] // 2),		# two samples per correlation time
	                latency = np.nan,									# climatology says nothing about events
	                top = clim['q999'])									# cover 99.9% of all values
	cat_req = dict(cadence = max(1, int(ep['rise'].quantile(1 - COVER)) // 2),	# two samples per precursor rise
	               latency = int(ep['lead'].quantile(1 - COVER)) - 1,			# data arrive before the onset
	               top = ep['peak_value'].quantile(COVER))					# cover the precursor peak
	return clim_req, cat_req

def true_requirements(w, train, test):
	"""
	TSS ratio for every degraded version of A, and the loosest value that keeps KEEP of the full TSS
	"""
	a = w['a']
	full = tss_of(w, a, train, test)
	curves = dict(cadence = [(k, tss_of(w, with_cadence(a, k), train, test) / full) for k in CADENCES],
	              latency = [(k, tss_of(w, with_latency(a, k), train, test) / full) for k in LATENCIES],
	              top = [(v, tss_of(w, with_saturation(a, v), train, test) / full) for v in TOPS])
	true_req = {}
	for name, curve in curves.items():
		first_fail = next((i for i, (v, r) in enumerate(curve) if r < KEEP), len(curve))
		true_req[name] = curve[first_fail - 1][0] if first_fail else np.nan		# last value before the first failure
	return full, curves, true_req

def kept_share(w, train, test, full, req):
	"""
	Share of the full TSS that A keeps when it just meets each requirement
	"""
	a = w['a']
	degrade = dict(cadence = with_cadence, latency = with_latency, top = with_saturation)
	return {name: tss_of(w, degrade[name](a, req[name]), train, test) / full if np.isfinite(req[name]) else np.nan
	        for name in degrade}

STYLES = dict(climatology = dict(color = '0.45', ls = '--', label = 'climatological requirement (standard)'),
              catalogue = dict(color = '#b8322a', ls = '-', label = 'catalogue requirement (ours)'),
              true = dict(color = 'k', ls = ':', label = 'true requirement'))
XLABELS = dict(cadence = 'cadence [steps between measurements]', latency = 'latency [steps]',
               top = 'saturation level of A')

def plot(curves, reqs):
	"""
	One panel per requirement: share of the full TSS against the requirement, with the three requirements marked
	(to the right the instrument gets worse in every panel)
	"""
	fig, ax = plt.subplots(1, 3, figsize = (13, 4.2), sharey = True)
	for axis, name in zip(ax, curves):
		v, r = zip(*curves[name])
		axis.plot(v, r, color = '0.2', marker = 'o', ms = 6, label = 'A degraded: TSS / full TSS')
		axis.axhline(KEEP, ls = '-.', color = '0.6', lw = 0.9, label = f'{KEEP:.0%} of the full TSS')
		for method, req in reqs.items():
			if np.isfinite(req[name]):
				axis.axvline(req[name], lw = 1.5, **STYLES[method])
		axis.set_xlabel(XLABELS[name])
		axis.set_title('   '.join(f'{m}: {req[name]:.3g}' if np.isfinite(req[name]) else f'{m}: n/a'
		                         for m, req in reqs.items()), fontsize = 9)	# coincident lines stay readable
	ax[2].invert_xaxis()
	ax[0].set_ylabel('share of the full TSS kept')
	ax[0].legend(loc = 'lower left', fontsize = 8)
	fig.tight_layout()
	fig.savefig(RES / 'run_2.pdf')
	return fig

if __name__ == '__main__':
	w = simulate(100_000, PARAMS)
	train, test = split(w)
	clim = climatology(w['a'])
	ep = episodes(w['a'], w['event'])
	clim_req, cat_req = requirements(clim, ep)
	full, curves, true_req = true_requirements(w, train, test)
	reqs = dict(climatology = clim_req, catalogue = cat_req, true = true_req)
	print(f'climatology: correlation time {clim["correlation_time"]} steps, 99.9% quantile {clim["q999"]:.2f}')
	print(f'catalogue: {len(ep)} episodes, percentiles\n{ep.quantile([0.1, 0.5, 0.9]).round(2)}')
	table = pd.DataFrame(reqs).T
	kept = pd.DataFrame({m: kept_share(w, train, test, full, r) for m, r in reqs.items()}).T.add_suffix(' kept')
	table = pd.concat([table, kept], axis = 1)
	print(f'\nfull TSS of A: {full:.3f}')
	print('requirements (cadence, latency: largest allowed; top: lowest allowed) and share of the full TSS kept')
	print(table.round(3))
	table.to_csv(RES / 'run_2_requirements.csv')
	plot(curves, reqs)
	print(f'\nsaved {RES / "run_2_requirements.csv"}, {RES / "run_2.pdf"}')
