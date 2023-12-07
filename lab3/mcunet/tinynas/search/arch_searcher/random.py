class RandomSearcher:
    def __init__(self, efficiency_predictor, accuracy_predictor, **kwargs):
        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor

        # evolution hyper-parameters
        self.max_time_budget = kwargs.get("max_time_budget", 500)

    @property
    def arch_manager(self):
        return self.accuracy_predictor.arch_encoder

    def update_hyper_params(self, new_param_dict):
        self.__dict__.update(new_param_dict)

    def random_valid_sample(self, constraint):
        while True:
            sample = self.arch_manager.random_sample_arch()
            efficiency = self.efficiency_predictor.get_efficiency(sample)
            if self.efficiency_predictor.satisfy_constraint(efficiency, constraint):
                return sample, efficiency

    def run_search(self, constraint, verbose=False, **kwargs):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        self.update_hyper_params(kwargs)

        child_pool = []
        efficiency_pool = []
        best_info = None
        if verbose:
            print("Generate random population...")
        for _ in tqdm(range(self.max_time_budget)):
            sample, efficiency = self.random_valid_sample(constraint)
            child_pool.append(sample)
            efficiency_pool.append(efficiency)

        accs = self.accuracy_predictor.predict_acc(child_pool)
        best_idx = accs.argmax()
        best_info = (accs.max(), child_pool[best_idx])
        return best_info