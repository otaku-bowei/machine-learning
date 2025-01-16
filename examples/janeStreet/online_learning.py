import time

import lightgbm


class LGBMTimeoutCallback:
    def __init__(self, timeout=None):
        self.timeout = timeout
        self.t0 = time.time_ns()
    def __call__(self, env):
        dt = (time.time_ns() - self.t0) / 1e9
        if self.timeout is not None and dt >= self.timeout:
            raise lightgbm.EarlyStopException(env.iteration,  env.evaluation_result_list)


params = {}


dataset = lightgbm.Dataset(data=x, label=y, free_raw_data=False)
model = None
for _ in range(10):
    model = lightgbm.train(
        params=params,
        train_set=dataset,
        num_boost_round=10,
        init_model=model,
        keep_training_booster=True
    )



class LGBMDataset(lightgbm.Dataset):
    def _set_init_score_by_predictor(self, predictor, data, used_indices):
        if self.init_score is None:
            return super()._set_init_score_by_predictor(predictor, data, used_indices)
        return self


dataset = LGBMDataset(data=x, label=y, free_raw_data=False)
model = None
for _ in range(10):
    num_trees_before = 0 if model is None else model.num_trees()
    model = lightgbm.train(
        params=params,
        train_set=dataset,
        num_boost_round=10,
        init_model=model,
        keep_training_booster=True
    )
    init_score = model.predict(dataset.data, start_iteration=num_trees_before)
    if dataset.init_score is not None:
        init_score = dataset.init_score + init_score.reshape(dataset.init_score.shape)
    dataset.set_init_score(init_score)
