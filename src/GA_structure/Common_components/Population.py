class Population:
    generation_nr = 0  # 静态变量，记录代数

    def __init__(self):
        self.fitness_score = None  # 整体种群的适应度（可选）
        self.individuals = {}  # 键为个体ID，值为个体对象
        self.generation_nr = Population.generation_nr
        Population.generation_nr += 1

    def create_first_population(self, size_pop, ind_factory):
        """
        生成初始种群个体。

        :param size_pop: 种群数量
        :param ind_factory: 个体生成工厂，需实现 create_random_individual()
        """
        print(f"开始创建初始种群，目标数量：{size_pop}")
        for i in range(size_pop):
            try:
                ind = ind_factory.create_random_individual()
                self.individuals[ind.id] = ind
                # print(f"[个体{i}] 创建成功")
                # print(f"  - ID: {ind.id}")
                # print(f"  - 参数: {ind.phenotype.params}")
                # print(f"  - 适应度: {ind.fitness_score}")
                # print(f"  - 可行性: {'✔️' if ind.phenotype.is_feasible() else '❌'}")
                # print("-" * 60)
            except Exception as e:
                print(f"[个体{i}] 创建失败 : {e}")

        print(f" 种群创建完成，共 {len(self.individuals)} 个体。")
        
        # for _ in range(size_pop):
        #     ind = ind_factory.create_random_individual()
        #     self.individuals[ind.id] = ind
           

    def get_size(self):
        """
        返回当前种群的个体数量
        """
        return len(self.individuals)

    def add_individual(self, individual):
        """
        向种群中添加一个个体
        """
        self.individuals[individual.id] = individual

    def get_population_fitness(self, feasible=True):
        """
        返回整个种群的适应度总和（仅当个体可行时）

        :param feasible: 若为 True，仅统计可行解（本版本暂未用）
        """
        return sum(individual.fitness_score for individual in self.individuals.values())

    def get_most_fitted_individual(self, fitness_obj, feasible=True):
        max_f = -999999999
        max_ind = None

        fallback_max_f = -999999999
        fallback_ind = None

        for id, individual in self.individuals.items():
            score = individual.fitness_score

            if score is None:
                continue

            #  可行性检查
            if fitness_obj.test_solution_is_feasible(individual):
                if score > max_f:
                    max_f = score
                    max_ind = individual
            else:
                # 记录一个不可行解中的最好者
                if score > fallback_max_f:
                    fallback_max_f = score
                    fallback_ind = individual

        #  返回策略
        if max_ind is not None:
            print("当前种群中有可行解，已返回最好的可行解个体。")
            return max_ind
        else:
            print("当前种群中无可行解，返回最好的不可行解个体。")
            return fallback_ind

