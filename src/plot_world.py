from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from world import PARAMS, simulate

RES = Path(__file__).parent.parent / 'results'

def corr_lag(u, v, k):
	"""
	correlation btw u[t], v[t+k]
	"""
	return np.corrcoef(u[:len(u) - k], v[k:])[0,1]
	
def main():
	p = PARAMS
	w = simulate(100_000, p)
	t0 = np.flatnonzero(w['event'])[50] - 100		# window starts 100 steps before the 51st event step
	t1 = t0 + 500
	t = np.arange(t0, t1)
	ev = w['event'][t0:t1] == 1     # steps with event in the window

	
	fig, ax = plt.subplots(3, 1, figsize = (9, 9))
	
	# source, environment, threshold and events
	
	ax[0].plot(t, w['d'][t0:t1], label = 'd (source)')
	ax[0].plot(t, w['x'][t0:t1], label = 'x (environment)')
	ax[0].axhline(w["theta"], ls="--", color="k", label = r'$\theta$ (threshold)')
	ax[0].plot(t[ev], w["x"][t0:t1][ev], "r.", label="event")
	ax[0].set_ylabel("value")
	ax[0].legend()
	
	# labels 
	
	ax[1].step(t, w["event"][t0:t1], where="post", label="event")         
	ax[1].step(t, w["y"][t0:t1] + 1.1, where="post", label="y (+1.1)")
	ax[1].set_ylabel("label")
	ax[1].legend()
	
	# correlations btw a[t] and x[t + k]
	lags = np.arange(0, 41)
	r = [corr_lag(w["a"], w["x"], k) for k in lags]                       # TODO 4
	ax[2].plot(lags, r, marker=".")
	ax[2].axvline(p["L"], ls="--", color="k", label="L")                  # TODO 5
	ax[2].set_xlabel("lag k [steps]")
	ax[2].set_ylabel("corr(a[t], x[t+k])")
	ax[2].legend()
	
	fig.tight_layout()
	fig.savefig(RES / 'world_check.pdf')
	
	print(f"saved {RES / 'world_check.pdf'}")
	

if __name__ == '__main__':
	main()
