aigs = [
    "apex1",
    "C1355",
    "C6288",
    "dalu",
    "k2",
    "bc0",
    "C5315",
    "C7552",
    "i10",
    "mainpla",
]

tot_job = 0
for seeds in range(10):
    for aig in aigs:
        tot_job += 1
        with open(f"jobs-graphtester/job_{tot_job}.sh", "w") as f:
            f.write(
                f"python3 aig_wl_analysis.py dataset-{aig}-all-actions-False-mc-simu-100-seed-{seeds}-rs-False"
            )
print(tot_job)
