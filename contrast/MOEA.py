import argparse
import os
import sys
import time
import csv
import numpy as np
import random

# 确保能导入同级的 momea 包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from momea.instance import Instance
try:
    from momea.runner import save_experiment_config, save_pareto_front, save_batch_summary
except Exception:
    save_experiment_config = None
    save_pareto_front = None
    save_batch_summary = None

# -----------------------------
# 全局参数（会在加载实例后设置）
# -----------------------------
num_jobs = 0
num_stages = 0
machines_per_stage = []
inst = None          # Instance 对象，用于按需计算时间和能耗

POP_SIZE = 50
GEN = 100
CROSS_RATE = 0.9
MUT_RATE = 0.2


# -----------------------------
# 解码函数（HFSP调度）
# -----------------------------
def decode(individual):
    """Individual 包含两个部分：
    - order: 作业顺序列表
    - speeds: 形状 (num_jobs,num_stages) 的速度索引矩阵
    """
    order = individual['order']
    speeds = individual['speeds']

    machine_available = []
    for s in range(num_stages):
        machine_available.append([0] * machines_per_stage[s])

    job_finish = np.zeros((num_jobs, num_stages))
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

    makespan = max(job_finish[:, -1])
    # standby energy as in momea.solution
    total_capacity = sum(machines_per_stage) * makespan
    standby = inst.standby_energy * max(0.0, total_capacity - total_proc_time)
    return makespan, total_energy + standby


# -----------------------------
# 初始化种群
# -----------------------------
def init_population():
    pop = []
    speed_choices = len(inst.speeds)
    for _ in range(POP_SIZE):
        order = list(range(num_jobs))
        random.shuffle(order)
        # 每个作业每阶段随机一个速度索引
        speeds = np.random.randint(0, speed_choices, size=(num_jobs, num_stages))
        pop.append({'order': order, 'speeds': speeds})
    return pop


# -----------------------------
# 非支配排序
# -----------------------------
def fast_nondominated_sort(values):
    # Use an explicit dominates predicate to avoid subtle equality issues
    def dominates(a, b, eps=1e-12):
        # a, b are objective tuples (f1, f2), minimization
        return (a[0] <= b[0] + eps and a[1] <= b[1] + eps) and (a[0] < b[0] - eps or a[1] < b[1] - eps)

    N = len(values)
    S = [[] for _ in range(N)]
    n = [0] * N
    fronts = [[]]

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominates(values[p], values[q]):
                S[p].append(q)
            elif dominates(values[q], values[p]):
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

    # remove last empty front if present
    if fronts and not fronts[-1]:
        fronts.pop()

    return fronts


# -----------------------------
# 拥挤度
# -----------------------------
def crowding_distance(values, front):

    distance = [0] * len(front)

    for m in range(2):

        sorted_index = sorted(range(len(front)),
                              key=lambda x: values[front[x]][m])

        distance[sorted_index[0]] = float('inf')
        distance[sorted_index[-1]] = float('inf')

        for i in range(1, len(front) - 1):

            prev = values[front[sorted_index[i - 1]]][m]
            nextv = values[front[sorted_index[i + 1]]][m]

            distance[sorted_index[i]] += nextv - prev

    return distance


# -----------------------------
# 交叉（OX）
# -----------------------------
def crossover(p1, p2):
    # Order 交叉使用 OX；speeds 使用均匀交换
    if random.random() > CROSS_RATE:
        # 深拷贝个体
        return {'order': p1['order'].copy(), 'speeds': p1['speeds'].copy()}

    # order 交叉
    a, b = sorted(random.sample(range(num_jobs), 2))
    child_order = [-1] * num_jobs
    child_order[a:b] = p1['order'][a:b]

    ptr = 0
    for x in p2['order']:
        if x not in child_order:
            while child_order[ptr] != -1:
                ptr += 1
            child_order[ptr] = x

    # speeds 交叉：每个 (job,stage) 随机选父亲
    child_speeds = np.zeros((num_jobs, num_stages), dtype=int)
    for j in range(num_jobs):
        for s in range(num_stages):
            if random.random() < 0.5:
                child_speeds[j, s] = p1['speeds'][j, s]
            else:
                child_speeds[j, s] = p2['speeds'][j, s]

    return {'order': child_order, 'speeds': child_speeds}


# -----------------------------
# 变异（swap）
# -----------------------------
def mutation(ind):
    # 对order做交换变异，对速度矩阵随机改变
    if random.random() < MUT_RATE:
        a, b = random.sample(range(num_jobs), 2)
        ind['order'][a], ind['order'][b] = ind['order'][b], ind['order'][a]

    # 速度变异：每个位置少量概率调整到其他索引
    for j in range(num_jobs):
        for s in range(num_stages):
            if random.random() < MUT_RATE:
                ind['speeds'][j, s] = random.randrange(len(inst.speeds))

    return ind


# -----------------------------
# 邻域搜索算子（四种邻域结构）
# -----------------------------
def neighborhood_search(ind, n_type):
    """四种邻域算子：
    1. 作业顺序插入
    2. 作业顺序交换
    3. 速度矩阵翻转（单作业速度改变）
    4. 速度矩阵交换（两作业速度交换）
    """
    new_ind = {'order': ind['order'].copy(), 'speeds': ind['speeds'].copy()}
    
    if n_type == 1:
        # 作业顺序插入
        if num_jobs > 1:
            i, j = random.sample(range(num_jobs), 2)
            job = new_ind['order'].pop(i)
            new_ind['order'].insert(j, job)
    elif n_type == 2:
        # 作业顺序交换
        if num_jobs > 1:
            i, j = random.sample(range(num_jobs), 2)
            new_ind['order'][i], new_ind['order'][j] = new_ind['order'][j], new_ind['order'][i]
    elif n_type == 3:
        # 速度矩阵翻转
        j = random.randint(0, num_jobs - 1)
        s = random.randint(0, num_stages - 1)
        new_ind['speeds'][j, s] = random.randrange(len(inst.speeds))
    elif n_type == 4:
        # 速度矩阵交换
        j1, j2 = random.sample(range(num_jobs), 2)
        s = random.randint(0, num_stages - 1)
        new_ind['speeds'][j1, s], new_ind['speeds'][j2, s] = new_ind['speeds'][j2, s], new_ind['speeds'][j1, s]
    
    return new_ind


# -----------------------------
# 基于作业的交叉算子
# -----------------------------
def job_based_crossover(p1, p2):
    """基于作业的交叉算子：order 使用 OX，speeds 使用均匀交叉"""
    # Order 交叉 (OX)
    a, b = sorted(random.sample(range(num_jobs), 2))
    child_order = [-1] * num_jobs
    child_order[a:b] = p1['order'][a:b]
    
    ptr = 0
    for x in p2['order']:
        if x not in child_order:
            while child_order[ptr] != -1:
                ptr += 1
            child_order[ptr] = x
    
    # Speeds 交叉：均匀交叉
    child_speeds = np.zeros((num_jobs, num_stages), dtype=int)
    for j in range(num_jobs):
        for s in range(num_stages):
            child_speeds[j, s] = p1['speeds'][j, s] if random.random() < 0.5 else p2['speeds'][j, s]
    
    return {'order': child_order, 'speeds': child_speeds}


# -----------------------------
# MOMPCEA 算法类
# -----------------------------
class MOMPCEA:
    def __init__(self, pop_size=50, gen=100):
        self.pop_size = pop_size
        self.gen = gen
        self.pop_makespan = []
        self.pop_tec = []
        self.pop_weighted = []
        self.ndss = []  # 非支配解集
        
        # 权重向量用于加权子种群
        self.weights = [[k/pop_size, 1 - k/pop_size] for k in range(pop_size)]
    
    def initialize_populations(self):
        """初始化三个子种群"""
        speed_choices = len(inst.speeds)
        
        for _ in range(self.pop_size):
            # Makespan 子种群：随机初始化
            order = list(range(num_jobs))
            random.shuffle(order)
            speeds = np.random.randint(0, speed_choices, size=(num_jobs, num_stages))
            self.pop_makespan.append({'order': order, 'speeds': speeds})
            
            # TEC 子种群：随机初始化
            order = list(range(num_jobs))
            random.shuffle(order)
            speeds = np.random.randint(0, speed_choices, size=(num_jobs, num_stages))
            self.pop_tec.append({'order': order, 'speeds': speeds})
            
            # Weighted 子种群：随机初始化
            order = list(range(num_jobs))
            random.shuffle(order)
            speeds = np.random.randint(0, speed_choices, size=(num_jobs, num_stages))
            self.pop_weighted.append({'order': order, 'speeds': speeds})
    
    def self_evolution(self):
        """自进化阶段：每个子种群使用基于作业的交叉算子"""
        # Makespan 子种群进化
        self._evolve_subpopulation(self.pop_makespan, lambda sol: decode(sol)[0])
        
        # TEC 子种群进化
        self._evolve_subpopulation(self.pop_tec, lambda sol: decode(sol)[1])
        
        # Weighted 子种群进化：使用不同的邻域结构和自适应机制
        self._evolve_weighted_subpopulation()
    
    def _evolve_subpopulation(self, population, fitness_func):
        """通用子种群进化：基于作业交叉 + 精英选择"""
        # 选择父代
        sorted_pop = sorted(population, key=fitness_func)
        parents = sorted_pop[:self.pop_size // 2]  # 精英选择
        
        offspring = []
        while len(offspring) < self.pop_size:
            p1, p2 = random.sample(parents, 2)
            child = job_based_crossover(p1, p2)
            # 变异
            if random.random() < MUT_RATE:
                child = mutation(child)
            offspring.append(child)
        
        # 精英保留
        population[:] = sorted_pop[:self.pop_size // 2] + offspring[:self.pop_size // 2]
    
    def _evolve_weighted_subpopulation(self):
        """加权子种群进化：使用邻域结构和自适应机制"""
        new_population = []
        
        for i, parent in enumerate(self.pop_weighted):
            w = self.weights[i % len(self.weights)]
            
            # 生成多个邻域解
            candidates = [parent]
            for n_type in range(1, 5):
                neighbor = neighborhood_search(parent, n_type)
                candidates.append(neighbor)
            
            # 选择最佳候选
            best_candidate = min(candidates, key=lambda sol: w[0] * decode(sol)[0] + w[1] * decode(sol)[1])
            new_population.append(best_candidate)
        
        self.pop_weighted = new_population
    
    def information_interaction(self):
        """三种信息共享战略"""
        # 战略1：精英个体交换
        self._exchange_elite_individuals()
        
        # 战略2：加权子种群多样性注入
        self._inject_weighted_diversity()
        
        # 战略3：加权子种群引导
        self._guide_weighted_subpopulation()
        
        # 更新 NDSS
        self._update_ndss()
    
    def _exchange_elite_individuals(self):
        """精英个体交换：makespan 和 TEC 子种群间交换精英"""
        self.pop_makespan.sort(key=lambda sol: decode(sol)[0])
        self.pop_tec.sort(key=lambda sol: decode(sol)[1])
        
        elite_count = max(1, self.pop_size // 10)
        elite_mk = self.pop_makespan[:elite_count]
        elite_tec = self.pop_tec[:elite_count]
        
        # 替换最差个体
        self.pop_tec[-elite_count:] = elite_mk
        self.pop_makespan[-elite_count:] = elite_tec
    
    def _inject_weighted_diversity(self):
        """注入加权子种群多样性到其他子种群"""
        num_inject = max(1, self.pop_size // 15)
        candidates = random.sample(self.pop_weighted, min(num_inject, len(self.pop_weighted)))
        
        for sol in candidates:
            if random.random() < 0.5:
                self.pop_makespan[-1] = sol.copy()
            else:
                self.pop_tec[-1] = sol.copy()
    
    def _guide_weighted_subpopulation(self):
        """使用精英信息引导加权子种群"""
        elites = self.ndss[:max(1, len(self.ndss) // 5)]
        diversity = random.sample(self.pop_makespan + self.pop_tec, min(len(self.pop_weighted) // 5, len(self.pop_makespan + self.pop_tec)))
        
        for i, elite in enumerate(elites):
            if i < len(self.pop_weighted):
                self.pop_weighted[i] = elite.copy()
        
        offset = len(elites)
        for j, div in enumerate(diversity):
            if offset + j < len(self.pop_weighted):
                self.pop_weighted[offset + j] = div.copy()
    
    def _update_ndss(self):
        """更新非支配解集"""
        all_solutions = self.pop_makespan + self.pop_tec + self.pop_weighted
        obj_values = [decode(sol) for sol in all_solutions]
        
        fronts = fast_nondominated_sort(obj_values)
        if fronts:
            # debug: print NDSS size for tracing
            try:
                print(f"_update_ndss: total_solutions={len(all_solutions)}, front0_size={len(fronts[0])}")
            except Exception:
                pass
            self.ndss = [all_solutions[i] for i in fronts[0]]
    
    def dynamic_vns(self):
        """动态可变邻域搜索：结合四种邻域算子"""
        if not self.ndss:
            return
        
        new_ndss = []
        neighborhood_order = [1, 2, 3, 4]
        
        for sol in self.ndss:
            current_sol = sol.copy()
            k = 0
            
            while k < len(neighborhood_order):
                current_neighborhood = neighborhood_order[k]
                best_local = current_sol
                improved = False
                
                # 在当前邻域中进行局部搜索
                for _ in range(10):  # 局部搜索迭代次数
                    neighbor = neighborhood_search(current_sol, current_neighborhood)
                    neighbor_obj = decode(neighbor)
                    current_obj = decode(current_sol)
                    
                    if (neighbor_obj[0] <= current_obj[0] and neighbor_obj[1] <= current_obj[1]) and \
                       (neighbor_obj[0] < current_obj[0] or neighbor_obj[1] < current_obj[1]):
                        best_local = neighbor
                        improved = True
                
                if improved:
                    current_sol = best_local
                    k = 0  # 重置到第一个邻域
                else:
                    k += 1
            
            new_ndss.append(current_sol)
        
        self.ndss = new_ndss
    
    def run(self):
        """运行 MOMPCEA 算法"""
        self.initialize_populations()
        
        for gen in range(self.gen):
            self.self_evolution()
            self.information_interaction()
            
            if gen % 10 == 0:
                print(f"Generation {gen}: NDSS size = {len(self.ndss)}")
        
        self.dynamic_vns()
        return self.ndss


# -----------------------------
# 主算法（修改为 MOMPCEA）
# -----------------------------
def MOEA_HFSP():
    algorithm = MOMPCEA(pop_size=POP_SIZE, gen=GEN)
    return algorithm.run()


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
    pop = MOEA_HFSP()
    run_time = time.time() - t0

    # If the algorithm already returned an NDSS (list of individuals),
    # avoid re-running nondominated sort which may cause unintended
    # duplicate/precision-based coalescing. Detect by structure (dict-based individual).
    if isinstance(pop, list) and pop and isinstance(pop[0], dict):
        pareto = pop
        pareto_objs = [decode(ind) for ind in pareto]
    else:
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
                save_experiment_config(os.path.join(save_results_dir, f"{base}_config.json"), algorithm='MOEA-HFSP', pop_size=POP_SIZE, gen=GEN)
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
            save_batch_summary(os.path.join(save_results_dir, 'batch_summary.json'), batch_results, {'algorithm': 'MOEA-HFSP'})
        except Exception:
            pass

    return batch_results


# -----------------------------
# 实例加载与主程序入口
# -----------------------------
def load_data(file_path):
    global num_jobs, num_stages, machines_per_stage, inst

    inst = Instance(file_path=file_path)
    num_jobs = inst.n
    num_stages = inst.m
    machines_per_stage = inst.machines

    print(f"Loaded instance {file_path}: {num_jobs} jobs, {num_stages} stages")


def main():
    parser = argparse.ArgumentParser(description="MOEA HFSP scheduler")
    parser.add_argument("--file", "-f", help="Single instance file path")
    parser.add_argument("--batch-dir", help="Directory containing instance files for batch run")
    parser.add_argument("--output-csv", default="batch_results.csv", help="Output CSV file for batch results")
    parser.add_argument("--results-dir", help="Directory to save detailed results")
    parser.add_argument("--pop-size", type=int, default=50, help="Population size")
    parser.add_argument("--gen", type=int, default=100, help="Number of generations")
    args = parser.parse_args()

    if args.batch_dir:
        # Batch run
        print(f"Starting MOEA-HFSP batch run on directory: {args.batch_dir}")
        run_batch(instances_dir=args.batch_dir, output_csv=args.output_csv, save_results_dir=args.results_dir, pop_size=args.pop_size, gen=args.gen)
    elif args.file:
        # Single run
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"Instance file not found: {args.file}")
        print(f"Running MOEA-HFSP on single instance: {args.file}")
        pop, pareto_objs, stats = run_experiment(args.file, pop_size=args.pop_size, gen=args.gen, save_results_dir=args.results_dir)
        print(f"Completed in {stats['run_time']:.2f}s - NDSS: {stats['ndss_size']}, Best: Cmax={stats['min_makespan']:.2f}, TEC={stats['min_tec']:.2f}")
    else:
        parser.error("Either --file or --batch-dir must be provided")


if __name__ == "__main__":
    main()