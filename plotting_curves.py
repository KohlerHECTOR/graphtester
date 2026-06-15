import matplotlib.pyplot as plt
import numpy as np

a = np.load('generlization_curves.npy')
m = np.mean(a, axis=0)
v = np.std(a, axis=0)
for aig in ["i10", "apex1", "C1355", "C6288", "dalu","k2", "bc0", "C5315", "C7552", "mainpla",]:
    b = np.load(f'{aig}_curve.npy')
    print(len(b), len(m))
    # plt.plot(m, linestyle='dotted', label='all AIGs')
    # plt.fill_between(np.arange(len(m)), m-v, m+v, alpha=0.2)
    plt.plot(b, label='single AIG')
    # plt.yscale('log')
    plt.legend()
    plt.savefig(f'plot_{aig}.pdf', dpi=300)
    plt.clf()

plt.plot(m, linestyle='dotted', label='all AIGs')
plt.fill_between(np.arange(len(m)), m-v, m+v, alpha=0.2)
plt.savefig('general.pdf')