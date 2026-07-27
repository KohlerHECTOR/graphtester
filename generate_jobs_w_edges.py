aigs = [
    "C6288",
]

tot_job = 0
for seeds in range(10):
    for aig in aigs:
        tot_job += 1
        with open(f"jobs-graphtester-w-edges/job_{tot_job}.sh", "w") as f:
            f.write(
                f"python3 aig_wl_analysis.py dataset-{aig}-all-actions-False-mc-simu-10-seed-{seeds}-rs-False --use-edges"
            )
print(tot_job)
