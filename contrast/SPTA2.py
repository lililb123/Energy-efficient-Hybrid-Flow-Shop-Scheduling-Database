import argparse
import os
import sys
import time
import csv
import numpy as np
import random

# Ensure momea package can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from momea.instance import Instance
try:
    from momea.runner import save_experiment_config, save_pareto_front, save_batch_summary
except Exception:
    save_experiment_config = None
    save_pareto_front = None
    save_batch_summary = None

# global variables initialized by load_data
num_jobs = 0
num_stages = 0
machines_per_stage = []
inst = None

# heuristic parameters
SPEED_BALANCE_WEIGHT = 0.5


# =================================
# 2 Decode solution
# =================================

def decode(individual):
    order = individual['order']
    speeds = individual['speeds']

    machine_available = []
    for s in range(num_stages):
        machine_available.append([0] * machines_per_stage[s])

    job_finish = np.zeros((num_jobs, num_stages), dtype=float)
    total_energy = 0.0
    total_proc_time = 0.0

    for job in order:
        for stage in range(num_stages):
            earliest_machine = 0
            earliest_time = float('inf')
            for m in range(machines_per_stage[stage]):
                start = max(
                    machine_available[stage][m],
                    job_finish[job][stage - 1] if stage > 0 else 0
                )
                if start < earliest_time:
                    earliest_time = start
                    earliest_machine = m

            speed_idx = speeds[job][stage]
            pt = inst.get_proc_time(job, stage, speed_idx)
            en = inst.get_proc_energy(job, stage, speed_idx)
            finish = earliest_time + pt

            machine_available[stage][earliest_machine] = finish
            job_finish[job][stage] = finish
            total_energy += en
            total_proc_time += pt

    makespan = float(np.max(job_finish[:, -1]))
    total_capacity = sum(machines_per_stage) * makespan
    standby = inst.standby_energy * max(0.0, total_capacity - total_proc_time)
    return makespan, total_energy + standby


# =================================
# 3 SPTA2 heuristic
# =================================

def generate_spta2_order():
    """Order jobs by total base processing time (SPT)."""
    job_times = [float(np.sum(inst.base_pt[j, :])) for j in range(num_jobs)]
    return sorted(range(num_jobs), key=lambda j: job_times[j])


def choose_speed_index(job, stage):
    """Choose a balanced speed index for the given operation."""
    speed_count = len(inst.speeds)
    pt_values = [inst.get_proc_time(job, stage, s) for s in range(speed_count)]
    energy_values = [inst.get_proc_energy(job, stage, s) for s in range(speed_count)]

    pt_max = max(pt_values)
    energy_max = max(energy_values)
    pt_max = pt_max if pt_max > 0 else 1.0
    energy_max = energy_max if energy_max > 0 else 1.0

    best_index = 0
    best_score = float('inf')
    for s in range(speed_count):
        pt_norm = pt_values[s] / pt_max
        energy_norm = energy_values[s] / energy_max
        score = SPEED_BALANCE_WEIGHT * pt_norm + (1.0 - SPEED_BALANCE_WEIGHT) * energy_norm
        if score < best_score:
            best_score = score
            best_index = s
    return best_index


def build_spta2_solution():
    order = generate_spta2_order()
    speeds = np.zeros((num_jobs, num_stages), dtype=int)

    machine_available = []
    for s in range(num_stages):
        machine_available.append([0] * machines_per_stage[s])

    job_finish = np.zeros((num_jobs, num_stages), dtype=float)

    for job in order:
        for stage in range(num_stages):
            earliest_machine = 0
            earliest_time = float('inf')
            for m in range(machines_per_stage[stage]):
                start = max(
                    machine_available[stage][m],
                    job_finish[job][stage - 1] if stage > 0 else 0
                )
                if start < earliest_time:
                    earliest_time = start
                    earliest_machine = m

            speed_idx = choose_speed_index(job, stage)
            pt = inst.get_proc_time(job, stage, speed_idx)
            finish = earliest_time + pt

            machine_available[stage][earliest_machine] = finish
            job_finish[job][stage] = finish
            speeds[job, stage] = speed_idx

    return {'order': order, 'speeds': speeds}


# =================================
# 4 Utility functions
# =================================

def _extract_pareto(pop):
    results = [decode(ind) for ind in pop]
    fronts = fast_nondominated_sort(results)
    pareto_idx = fronts[0]
    pareto = [pop[i] for i in pareto_idx]
    pareto_objs = [results[i] for i in pareto_idx]
    return pareto, pareto_objs


def fast_nondominated_sort(values):
    S = [[] for _ in range(len(values))]
    n = [0] * len(values)
    fronts = [[]]

    for p in range(len(values)):
        for q in range(len(values)):
            if p == q:
                continue
            if (values[p][0] <= values[q][0] and values[p][1] <= values[q][1] and values[p] != values[q]):
                S[p].append(q)
            elif (values[q][0] <= values[p][0] and values[q][1] <= values[p][1] and values[p] != values[q]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    if fronts and not fronts[-1]:
        fronts.pop()
    return fronts


# =================================
# 5 Experiment and batch helpers
# =================================

def run_experiment(file_path: str, save_results_dir: str = None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    load_data(file_path)
    t0 = time.time()
    sol = build_spta2_solution()
    run_time = time.time() - t0

    pop = [sol]
    pareto, pareto_objs = _extract_pareto(pop)

    makespans = [o[0] for o in pareto_objs]
    tecs = [o[1] for o in pareto_objs]

    stats = {
        'run_time': run_time,
        'ndss_size': len(pareto),
        'min_makespan': min(makespans) if makespans else None,
        'min_tec': min(tecs) if tecs else None,
    }

    if save_results_dir:
        os.makedirs(save_results_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(file_path))[0]
        if save_experiment_config:
            try:
                save_experiment_config(os.path.join(save_results_dir, f"{base}_config.json"), algorithm='SPTA2', pop_size=1, gen=1)
            except Exception:
                pass
        if save_pareto_front:
            try:
                save_pareto_front(os.path.join(save_results_dir, f"{base}_pareto.json"), pareto, base)
            except Exception:
                try:
                    import json
                    with open(os.path.join(save_results_dir, f"{base}_pareto.json"), 'w', encoding='utf-8') as f:
                        json.dump({'pareto_objs': pareto_objs}, f, indent=2)
                except Exception:
                    pass

    return pop, pareto_objs, stats


def run_batch(instances_dir: str = None, file_list: list = None, output_csv: str = 'batch_results.csv', save_results_dir: str = None):
    files = []
    if file_list:
        files = [f for f in file_list if os.path.exists(f)]
    elif instances_dir and os.path.exists(instances_dir):
        files = sorted([os.path.join(instances_dir, f) for f in os.listdir(instances_dir) if f.endswith('.txt')])
    else:
        raise ValueError('No valid instances_dir or file_list provided')

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    if save_results_dir:
        os.makedirs(save_results_dir, exist_ok=True)

    batch_results = []
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['instance_file', 'n', 'm', 'makespan', 'tec', 'ndss_size', 'run_time', 'status'])

        for fpath in files:
            entry = {'instance_file': os.path.basename(fpath), 'status': 'pending'}
            try:
                base = os.path.splitext(os.path.basename(fpath))[0]
                results_dir = os.path.join(save_results_dir, base) if save_results_dir else None
                pop, pareto_objs, stats = run_experiment(fpath, save_results_dir=results_dir)

                entry.update({
                    'n': inst.n,
                    'm': inst.m,
                    'min_makespan': stats['min_makespan'],
                    'min_tec': stats['min_tec'],
                    'ndss_size': stats['ndss_size'],
                    'run_time': stats['run_time'],
                    'status': 'completed'
                })

                for o in pareto_objs:
                    writer.writerow([os.path.basename(fpath), inst.n, inst.m, f"{o[0]:.6f}", f"{o[1]:.6f}", stats['ndss_size'], f"{stats['run_time']:.4f}", 'completed'])

            except Exception as e:
                entry.update({'status': 'failed', 'error': str(e), 'run_time': 0})
                writer.writerow([os.path.basename(fpath), 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', '0', 'failed'])

            batch_results.append(entry)

    if save_results_dir and save_batch_summary:
        try:
            save_batch_summary(os.path.join(save_results_dir, 'batch_summary.json'), batch_results, {'algorithm': 'SPTA2'})
        except Exception:
            pass

    return batch_results


# =================================
# 6 Instance loading
# =================================

def load_data(file_path):
    global num_jobs, num_stages, machines_per_stage, inst
    inst = Instance(file_path=file_path)
    num_jobs = inst.n
    num_stages = inst.m
    machines_per_stage = inst.machines
    print(f"Loaded instance {file_path}: {num_jobs} jobs, {num_stages} stages")


def main():
    parser = argparse.ArgumentParser(description="SPTA2 scheduler")
    parser.add_argument("-f", "--file", help="Single instance file path")
    parser.add_argument("--batch-dir", help="Directory containing instance files for batch run")
    parser.add_argument("--output-csv", default="batch_results.csv", help="Output CSV file for batch results")
    parser.add_argument("--results-dir", help="Directory to save detailed results")
    args = parser.parse_args()

    if args.batch_dir:
        print(f"Starting SPTA2 batch run on directory: {args.batch_dir}")
        run_batch(instances_dir=args.batch_dir, output_csv=args.output_csv, save_results_dir=args.results_dir)
    elif args.file:
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"Instance file not found: {args.file}")
        print(f"Running SPTA2 on single instance: {args.file}")
        pop, pareto_objs, stats = run_experiment(args.file, save_results_dir=args.results_dir)
        print(f"Completed in {stats['run_time']:.2f}s - NDSS: {stats['ndss_size']}, Best: Cmax={stats['min_makespan']:.2f}, TEC={stats['min_tec']:.2f}")
    else:
        parser.error("Either --file or --batch-dir must be provided")


if __name__ == "__main__":
    main()
