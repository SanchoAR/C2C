"""
Unified OSE procedure: the same data-denial experiment for any forecast engine
"""

from ose_assim import assim_forecast, tss_max
from ose_stat import split, stat_forecast
from world import PARAMS, simulate

# forecast engines: each one is engine(w, instruments, train) -> event probability at every step
ENGINES = dict(
kalman = lambda w, inst, train: assim_forecast(w, inst, train, PARAMS),
logistic = lambda w, inst, train: stat_forecast(w, inst, train, 'logistic'),
boosting = lambda w, inst, train: stat_forecast(w, inst, train, 'boosting')
)

def denial(engine, w, train, test, instruments = ('a','b','c'), alone = True):
	"""
	TSS on the test steps with all instruments, without each one, and (if alone) with each one alone
	"""
	runs = {'all':instruments}
	for name in instruments:
		runs['without ' + name] = tuple(i for i in instruments if i != name)
	if alone:
		for name in instruments:
			runs['only ' + name] = (name,)
	
	tss = {}
	for label, inst in runs.items():
		prob = engine(w, inst, train)
		tss[label] = tss_max(w['y'][test], prob[test])
	return tss
	
if __name__ == '__main__':
	w = simulate(100_000, PARAMS)
	train, test = split(w)
	res = {name : denial(engine, w, train, test) for name, engine in ENGINES.items()}
	print(f"{'TSS':14s}" + ''.join(f'{name:>10s}' for name in res))
	for label in res['kalman']:
		print(f'{label:14s}' + ''.join(f'{res[name][label]:10.3f}' for name in res))
	print(f"{'loss':14s}")
	for i in ('a','b','c'):
		print(f'{"without " + i:14s}' + ''.join(f"{res[name]['all'] - res[name]['without ' + i]:10.3f}" for name in res))
