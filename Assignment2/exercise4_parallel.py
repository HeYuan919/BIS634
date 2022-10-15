import numpy as np
from multiprocessing import Process, Pool
import random
import time
import matplotlib.pyplot as plt

def alg1(data):
  data = list(data)
  changes = True
  while changes:
    changes = False
    for i in range(len(data) - 1):
      if data[i + 1] < data[i]:
        data[i], data[i + 1] = data[i + 1], data[i]
        changes = True
  return data

def alg2(data):
  if len(data) <= 1:
    return data
  else:
    split = len(data) // 2
    left = iter(alg2(data[:split]))
    right = iter(alg2(data[split:]))
    result = []
    # note: this takes the top items off the left and right piles
    left_top = next(left)
    right_top = next(right)
    while True:
      if left_top < right_top:
        result.append(left_top)
        try:
          left_top = next(left)
        except StopIteration:
          # nothing remains on the left; add the right + return
          return result + [right_top] + list(right)
      else:
        result.append(right_top)
        try:
          right_top = next(right)
        except StopIteration:
          # nothing remains on the right; add the left + return
          return result + [left_top] + list(left)

def data1(n, sigma=10, rho=28, beta=8/3, dt=0.01, x=1, y=1, z=1):
    import numpy
    state = numpy.array([x, y, z], dtype=float)
    result = []
    for _ in range(n):
        x, y, z = state
        state += dt * numpy.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])
        result.append(float(state[0] + 30))
    return result

if __name__ == "__main__":
    N = np.logspace(1, 6, 50, endpoint=True)
    timeused_1 = []
    timeused_2 = []
    pool = Pool(2)
    for n in N:
        data_gen = data1(int(n))

        startime = time.perf_counter()
        split = len(data_gen) // 2
        data_gen1 = data_gen[:split]
        data_gen2 = data_gen[split:]
        r1, r2 = pool.map(alg2,[data_gen1,data_gen2])
        iter1 = iter(r1)
        iter2 = iter(r2)
        result = []
        iter1_top = next(iter1)
        iter2_top = next(iter2)
        while True:
            if iter1_top < iter2_top:
                result.append(iter1_top)
                try:
                    iter1_top = next(iter1)
                except StopIteration:
                    result = result + [iter2_top] + list(iter2)
                    break
            else:
                result.append(iter2_top)
                try:
                    iter2_top = next(iter2)
                except StopIteration:
                    result = result + [iter1_top] + list(iter1)
                    break
        endtime = time.perf_counter()
        timeused = endtime - startime
        timeused_1.append(timeused)

        startime = time.perf_counter()
        alg2(data_gen)
        endtime = time.perf_counter()
        timeused = endtime - startime
        timeused_2.append(timeused)

    pool.close()
    pool.join()


    plt.axes(xscale='log', yscale='log')
    plt.xlabel('n')
    plt.ylabel('time_used')
    plt.plot(N, timeused_1, color='b', label='parallel-algr2')
    plt.plot(N, timeused_2, color='r', label='algr2')
    plt.title('data1')
    plt.legend()
    plt.savefig('parallelondata1.jpg')
    plt.show()

    quotient = [a/b for a, b in zip(timeused_1, timeused_2)]
    plt.axes(xscale='log', yscale='log')
    plt.xlabel('n')
    plt.ylabel('time_used')
    plt.plot(N,quotient,label='parallel/original')
    plt.axhline(1,color='gray',label='1')
    plt.axhline(0.5, color='red',label='0.5')
    plt.title('quotient')
    plt.legend()
    plt.savefig('quotient.jpg')
    plt.show()

    print('n=10^6,paralleled time is %f, original time is'%timeused_1[-1], timeused_2[-1])