import ydf
import numpy as np

from collections.abc import Mapping
from .mean_ydf_loss import mean_loss
from .quantile_ydf_loss import make_quantile_loss

from bayesflow.scores import ScoringRule, MeanScore, QuantileScore


class GradientBoostingWorkflow:
    def __init__(
        self,
        simulator,
        adapter,
        scores: dict[str, ScoringRule] = None,
        ydf_learner_class=ydf.GradientBoostedTreesLearner,
    ):
        self.simulator = simulator
        self.adapter = adapter
        self.scores = scores
        self.ydf_learner_class = ydf_learner_class

    def fit_offline(
        self,
        data: Mapping[str, np.ndarray],
        validation_data: Mapping[str, np.ndarray] | int = None,
        # augmentations: Mapping[str, Callable] | Callable = None,
        **kwargs,
    ):
        forest_data = self._prepare_data(data)
        forest_val_data = self._prepare_data(validation_data)

        self.models = {}
        for score_key, score in self.scores.items():
            if isinstance(score, MeanScore):
                self.models[score_key] = {
                    key: [
                        self.ydf_learner_class(label="target", task=ydf.Task.REGRESSION, loss=mean_loss).train(
                            forest_data[key], valid=forest_val_data[key], **self._filter_train(kwargs)
                        )
                    ]
                    for key in forest_data.keys()
                }
            elif isinstance(score, QuantileScore):
                self.models[score_key] = {
                    key: [
                        self.ydf_learner_class(
                            label="target", task=ydf.Task.REGRESSION, loss=make_quantile_loss(tau)
                        ).train(forest_data[key], valid=forest_val_data[key], **self._filter_train(kwargs))
                        for tau in score.q
                    ]
                    for key in forest_data.keys()
                }

    def _filter_train(self, kwargs):
        return {key: value for key, value in kwargs.items() if key in ["verbose"]}

    def estimate(
        self,
        *,
        conditions: Mapping[str, np.ndarray],
        **kwargs,
    ) -> dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]]:
        """
        Estimates point summaries of inference variables based on specified conditions.
        """

        forest_conditions = self._prepare_data(conditions)

        estimates = {}
        for score_key, score in self.scores.items():
            estimates[score_key] = {
                condition_key: [
                    self.models[score_key][condition_key][i].predict(forest_conditions[condition_key])
                    for i in range(len(self.models[score_key][condition_key]))
                ]
                for condition_key in forest_conditions.keys()
            }

        for score_key, score in self.scores.items():
            estimates[score_key] = self.adapter(
                dict(
                    inference_variables=np.transpose(list(estimates[score_key].values())),
                ),
                inverse=True,
            )
        estimates = GradientBoostingWorkflow._reorder_estimates(estimates)

        return estimates

    @staticmethod
    def _reorder_estimates(
        estimates: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    ) -> dict[str, dict[str, dict[str, np.ndarray]]]:
        """Reorders the nested dictionary so that the inference variable names become the top-level keys."""
        # Grab the variable names from one sample inner dictionary.
        sample_inner = next(iter(estimates.values()))
        variable_names = sample_inner.keys()
        reordered = {}
        for variable in variable_names:
            reordered[variable] = {}
            for score_key, inner_dict in estimates.items():
                reordered[variable][score_key] = estimates[score_key][variable]
                # inner_dict[variable] for inner_key, value in inner_dict.items()}
        return reordered

    def _prepare_data(self, data):
        data = self.adapter(data)

        output_dim = data["inference_variables"].shape[-1]

        responses = data.pop("inference_variables")
        responses = {f"x_{i}": v.squeeze() for i, v in enumerate(np.hsplit(responses, output_dim))}

        data |= responses

        return {
            key: {
                "features": data["inference_conditions"],
                "target": data[key],
            }
            for key in responses.keys()
        }
