import argparse
import os
import sys
import time
import csv
import numpy as np
import random

# for importing momea
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from momea.instance import Instance
try:
    from momea.runner import save_experiment_config, save_pareto_front, save_batch_summary
except Exception:
    save_experiment_config = None
    save_pareto_front = None
    save_batch_summary = None

# global params set by load_data
num_jobs = 0
num_stages = 0
machines_per_stage = []
inst = None

POP_SIZE = 60
GEN = 100
CROSS_RATE = 0.9
MUT_RATE = 0.2

NUM_OBJECTIVES = 3  # makespan, energy, load_balance


# =================================
# 2 HFSP 解码
# =================================

def decode(individual):
    order = individual['order']
    speeds = individual['speeds']

    machine_available = []
    for s in range(num_stages):
        machine_available.append([0]*machines_per_stage[s])

    job_finish = np.zeros((num_jobs,num_stages))
    machine_load = []
    for s in range(num_stages):
        machine_load.append([0]*machines_per_stage[s])

    total_energy = 0.0
    total_proc_time = 0.0

    for job in order:
        for stage in range(num_stages):
            best_machine = 0
            best_start = float("inf")

            for m in range(machines_per_stage[stage]):
                start = max(
                    machine_available[stage][m],
                    job_finish[job][stage-1] if stage>0 else 0
                )
                if start < best_start:
                    best_start = start
                    best_machine = m

            speed_idx = speeds[job][stage]
            pt = inst.get_proc_time(job, stage, speed_idx)
            en = inst.get_proc_energy(job, stage, speed_idx)

            finish = best_start + pt
            machine_available[stage][best_machine] = finish
            machine_load[stage][best_machine] += pt
            job_finish[job][stage] = finish

            total_energy += en
            total_proc_time += pt

    makespan = max(job_finish[:,-1])
    total_capacity = sum(machines_per_stage) * makespan
    standby = inst.standby_energy * max(0.0, total_capacity - total_proc_time)
    energy_obj = total_energy + standby

    loads = []
    for s in range(num_stages):
        loads += machine_load[s]
    load_balance = np.std(loads)

    return (makespan, energy_obj, load_balance)


# =================================
# 3 初始化
# =================================

def init_population():
    pop = []
    speed_choices = len(inst.speeds)
    for _ in range(POP_SIZE):
        order = list(range(num_jobs))
        random.shuffle(order)
        speeds = np.random.randint(0, speed_choices, size=(num_jobs, num_stages))
        pop.append({'order': order, 'speeds': speeds})
    return pop


# =================================
# 4 非支配排序
# =================================

def dominates(a,b):

    return all(x<=y for x,y in zip(a,b)) and any(x<y for x,y in zip(a,b))


def fast_nondominated_sort(values):

    S = [[] for _ in range(len(values))]
    n = [0]*len(values)

    fronts=[[]]

    for p in range(len(values)):

        for q in range(len(values)):

            if dominates(values[p],values[q]):
                S[p].append(q)

            elif dominates(values[q],values[p]):
                n[p]+=1

        if n[p]==0:
            fronts[0].append(p)

    i=0

    while fronts[i]:

        next_front=[]

        for p in fronts[i]:

            for q in S[p]:

                n[q]-=1

                if n[q]==0:
                    next_front.append(q)

        i+=1
        fronts.append(next_front)

    fronts.pop()

    return fronts


# =================================
# 5 参考点生成
# =================================

def generate_reference_points(M, divisions):

    ref_points=[]

    def generate_recursive(point,left,total,start):

        if start==M-1:

            point[start]=left/total
            ref_points.append(point.copy())

        else:

            for i in range(left+1):

                point[start]=i/total
                generate_recursive(point,left-i,total,start+1)

    generate_recursive([0]*M,divisions,divisions,0)

    return np.array(ref_points)


REF_POINTS = generate_reference_points(NUM_OBJECTIVES,4)


# =================================
# 6 选择
# =================================

def tournament(pop,obj):
    i,j=random.sample(range(len(pop)),2)
    if obj[i] < obj[j]:
        return pop[i]
    else:
        return pop[j]


# =================================
# 7 交叉
# =================================

def crossover(p1,p2):
    if random.random()>CROSS_RATE:
        return {'order': p1['order'].copy(), 'speeds': p1['speeds'].copy()}
    a,b=sorted(random.sample(range(num_jobs),2))
    child_order=[-1]*num_jobs
    child_order[a:b]=p1['order'][a:b]
    ptr=0
    for x in p2['order']:
        if x not in child_order:
            while child_order[ptr]!=-1:
                ptr+=1
            child_order[ptr]=x
    child_speeds=np.zeros((num_jobs,num_stages),dtype=int)
    for j in range(num_jobs):
        for s in range(num_stages):
            child_speeds[j,s] = p1['speeds'][j,s] if random.random()<0.5 else p2['speeds'][j,s]
    return {'order': child_order, 'speeds': child_speeds}


# =================================
# 8 变异
# =================================

def mutation(ind):
    if random.random()<MUT_RATE:
        i,j=random.sample(range(num_jobs),2)
        ind['order'][i], ind['order'][j] = ind['order'][j], ind['order'][i]
    for j in range(num_jobs):
        for s in range(num_stages):
            if random.random()<MUT_RATE:
                ind['speeds'][j,s] = random.randrange(len(inst.speeds))
    return ind


# =================================
# 9 Niching选择 (简化版)
# =================================

def niching_selection(front,ref_points,obj,needed):

    selected=[]

    points=np.array([obj[i] for i in front])

    normalized=(points - points.min(axis=0))/(points.max(axis=0)-points.min(axis=0)+1e-6)

    distance=np.linalg.norm(
        normalized[:,None,:]-ref_points[None,:,:],
        axis=2
    )

    association=np.argmin(distance,axis=1)

    ref_count={i:0 for i in range(len(ref_points))}

    while len(selected)<needed:

        for r in range(len(ref_points)):

            candidates=[i for i,a in enumerate(association) if a==r and front[i] not in selected]

            if candidates:

                best=min(candidates,key=lambda x:distance[x,r])

                selected.append(front[best])
                ref_count[r]+=1

                if len(selected)==needed:
                    break

    return selected


# =================================
# 10 NSGA-III
# =================================

def nsga3():

    population=init_population()

    for g in range(GEN):

        obj=[decode(ind) for ind in population]

        offspring=[]

        while len(offspring)<POP_SIZE:

            p1=tournament(population,obj)
            p2=tournament(population,obj)

            child=crossover(p1,p2)
            child=mutation(child)

            offspring.append(child)

        combined=population+offspring

        obj_combined=[decode(ind) for ind in combined]

        fronts=fast_nondominated_sort(obj_combined)

        new_population=[]

        for front in fronts:

            if len(new_population)+len(front)<=POP_SIZE:

                for idx in front:
                    new_population.append(combined[idx])

            else:

                needed=POP_SIZE-len(new_population)

                chosen=niching_selection(front,REF_POINTS,obj_combined,needed)

                for idx in chosen:
                    new_population.append(combined[idx])

                break

        population=new_population

    return population


def _extract_pareto(pop):
    results = [decode(ind) for ind in pop]
    fronts = fast_nondominated_sort(results)
    pareto_idx = fronts[0]
    pareto = [pop[i] for i in pareto_idx]
    pareto_objs = [results[i] for i in pareto_idx]
    return pareto, pareto_objs


def run_experiment(file_path: str, pop_size: int = None, gen: int = None, save_results_dir: str = None):
    global POP_SIZE, GEN
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    load_data(file_path)
    if pop_size is not None:
        POP_SIZE = pop_size
    if gen is not None:
        GEN = gen

    t0 = time.time()
    pop = nsga3()
    run_time = time.time() - t0

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
                save_experiment_config(os.path.join(save_results_dir, f"{base}_config.json"), algorithm='NSGA-III', pop_size=POP_SIZE, gen=GEN)
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


def run_batch(instances_dir: str = None, file_list: list = None, output_csv: str = 'batch_results.csv', save_results_dir: str = None, pop_size: int = None, gen: int = None):
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
                pop, pareto_objs, stats = run_experiment(fpath, pop_size=pop_size, gen=gen, save_results_dir=results_dir)

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
            save_batch_summary(os.path.join(save_results_dir, 'batch_summary.json'), batch_results, {'algorithm': 'NSGA-III'})
        except Exception:
            pass

    return batch_results


# =================================
# 运行与入口
# =================================

def load_data(file_path):
    global num_jobs, num_stages, machines_per_stage, inst
    inst = Instance(file_path=file_path)
    num_jobs = inst.n
    num_stages = inst.m
    machines_per_stage = inst.machines
    print(f"Loaded instance {file_path}: {num_jobs} jobs, {num_stages} stages")


def main():
    parser = argparse.ArgumentParser(description="NSGA-III scheduler")
    parser.add_argument("-f", "--file", help="Single instance file path")
    parser.add_argument("--batch-dir", help="Directory containing instance files for batch run")
    parser.add_argument("--output-csv", default="batch_results.csv", help="Output CSV file for batch results")
    parser.add_argument("--results-dir", help="Directory to save detailed results")
    parser.add_argument("--pop-size", type=int, default=60, help="Population size")
    parser.add_argument("--gen", type=int, default=100, help="Number of generations")
    args = parser.parse_args()

    if args.batch_dir:
        # Batch run
        print(f"Starting NSGA-III batch run on directory: {args.batch_dir}")
        run_batch(instances_dir=args.batch_dir, output_csv=args.output_csv, save_results_dir=args.results_dir, pop_size=args.pop_size, gen=args.gen)
    elif args.file:
        # Single run
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"Instance file not found: {args.file}")
        print(f"Running NSGA-III on single instance: {args.file}")
        pop, pareto_objs, stats = run_experiment(args.file, pop_size=args.pop_size, gen=args.gen, save_results_dir=args.results_dir)
        print(f"Completed in {stats['run_time']:.2f}s - NDSS: {stats['ndss_size']}, Best: Cmax={stats['min_makespan']:.2f}, TEC={stats['min_tec']:.2f}")
    else:
        parser.error("Either --file or --batch-dir must be provided")

if __name__ == "__main__":
    main()