import threading

import matplotlib.pyplot as plt
import pandas as pd
import scipy
import percolate


n_threads = 2


results = []


def print_numbers(start, end):
    for i in range(start, end + 1):
        print(i)


# Create a list of thread objects
threads = []

PATH = r"C:\Users\jonah\Desktop\recording_731M_Spontaneous_50Hz_Naive_no_injections.csv"

resolution = 8

'''sam_df = pd.read_csv(PATH, header=None, skiprows=40000, nrows=30)

time = []
avg = []

print(1)

t = 0
for col in sam_df:
    time.append(t)
    avg.append(abs(sam_df[col].mean()))

    t += 0.02

plt.plot(time, avg)
plt.show()'''

timepoints = [pd.DataFrame(columns=[i for i in range(resolution)], index=[i for i in range(resolution)]) for j in range(20)]

for i in range(5, resolution):
    for j in range(0, resolution):
        print(f"{i}, {j}")

        r = 8192 * i + 32 * j
        data = pd.read_csv(PATH, header=None, skiprows=r, nrows=1)

        for k, df in enumerate(timepoints):
            df.iloc[i].iloc[j] = data.iloc[0].iloc[6500 + (k * 2)]
            if j == resolution - 1:
                df.to_csv(f"t2\\{k}.csv", index=False)


print("Program completed.")
