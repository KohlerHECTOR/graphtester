aigs = [
    "C6288",
    "mainpla",
]

tot_job = 0
for seeds in range(10):
    for aig in aigs:
        tot_job += 1
        with open(f"jobs-graphtester-all-actions/job_{tot_job}.sh", "w") as f:
            f.write(
                f"python3 aig_wl_analysis.py dataset-{aig}-all-actions-True-mc-simu-5-seed-{seeds}-rs-False"
            )
print(tot_job)
