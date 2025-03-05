import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F


EPS = 1e-9


class HDRLoss(torch.nn.Module):

    def __init__(self, lambda_hdr):
        super().__init__()
        self.lambda_hdr = lambda_hdr
        self.criterion = DivDisLossWrapper(
            task_loss=torch.nn.CrossEntropyLoss(),
            weight=0.5,
            mode="mi",
            reduction="mean",
            mapper=None,
            loss_type="a2d",
            use_always_labeled=True,
            modifier="budget",
            gamma=2.0,
            disagree_after_epoch=0,
            manual_lambda=self.lambda_hdr,
            disagree_below_threshold=None,
            reg_mode=None,
            reg_weight=None
        )

    def forward(self, logits, targets):
        """Computes the HDRLoss.

        Args:
            logits: Input logits of shape [B, S, C], B is the batch size, S is the number of heads, and C is the number of classes.
            targets: Ground truth labels of shape [B].

        Returns:
            The computed HDRLoss.
        """
        num_heads = logits.shape[1]
        logits = torch.chunk(logits, num_heads, dim=1)
        logits = [logit.squeeze(1) for logit in logits]
        return self.criterion(logits, targets)


class DivDisLossWrapper(torch.nn.Module):

    def __init__(
        self,
        task_loss,
        weight,
        mode="mi",
        reduction="mean",
        mapper=None,
        loss_type="divdis",
        use_always_labeled=False,
        modifier=None,
        gamma=2.0,
        disagree_after_epoch=0,
        manual_lambda=1.0,
        disagree_below_threshold=None,
        reg_mode=None,
        reg_weight=None
    ):
        super().__init__()
        self.repulsion_loss = None
        self.mode = mode
        self.reduction = reduction
        self.task_loss = task_loss
        self.weight = weight
        self.loss_type = loss_type
        # if mapper == "mvh":
        #     self.mapper = make_mvh_mapper()
        # else:
        #     assert mapper is None, "Only mvh mapper is supported"
        #     self.mapper = None
        assert mapper is None, "mapper is not implemented"
        self.log_this_batch = True
        self.use_always_labeled = use_always_labeled
        self.modifier = modifier
        self.gamma = gamma
        self.epoch = 0
        self.disagree_after_epoch = disagree_after_epoch
        self.manual_lambda = manual_lambda
        self.disagree_below_threshold = disagree_below_threshold
        self.reg_mode = reg_mode
        self.reg_weight = reg_weight

        if self.reg_mode is not None:
            assert self.reg_weight is not None

    def increase_epoch(self):
        self.epoch += 1

    def compute_modifier(self, outputs, targets):

        if self.modifier == "budget":
            return budget_modifier(outputs, targets)
        else:
            assert self.modifier is None
            return 1.0

    def forward(self, outputs, targets, unlabeled_outputs=None):

        def zero_if_none(value):
            return (
                value.item()
                    if value is not None
                    else 0
            )

        def get_repulsion_loss(outputs, unlabeled_outputs, targets_values):
            if unlabeled_outputs is None:
                assert not outputs[0][1].requires_grad, \
                    "No unlabeled batch was provided during training"
                repulsion_loss = None
                modifier = None
            else:
                n_heads = len(unlabeled_outputs)
                if self.repulsion_loss is None:

                    if self.loss_type == "divdis":
                        assert False, "DivDisLoss is not implemented"
                        # assert self.modifier != "budget", \
                        #     "Budget modifier is not supported for DivDisLoss"
                        # self.repulsion_loss = DivDisLoss(
                        #     n_heads,
                        #     self.mode,
                        #     self.reduction
                        # )
                    else:
                        assert self.loss_type == "a2d"
                        reduction = "mean"
                        if self.modifier == "budget":
                            reduction = "none"
                        self.repulsion_loss = A2DLoss(
                            n_heads,
                            reduction=reduction
                        )
                else:
                    self.repulsion_loss.heads = n_heads

                if self.use_always_labeled:
                    modifier = self.compute_modifier(
                        unlabeled_outputs,
                        targets_values
                    )
                else:
                    assert self.modifier is None
                    modifier = 1.0

                # if self.mapper is not None:
                #     unlabeled_outputs \
                #         = [self.mapper(output) for output in unlabeled_outputs]

                # [batch, n * classes]
                unlabeled_outputs_cat = torch.cat(
                    unlabeled_outputs,
                    axis=-1
                )

                repulsion_loss = self.repulsion_loss(unlabeled_outputs_cat)
            return repulsion_loss, modifier, unlabeled_outputs

        if self.use_always_labeled:
            assert unlabeled_outputs is None
            unlabeled_outputs = outputs

        # cur_n_heads = len(outputs)

        # metrics_mappings = get_metrics_mapping(
        #     cur_n_heads > MAX_MODELS_WITHOUT_OOM
        # )

        # outputs = bootstrap_ensemble_outputs(outputs)
        # targets_values = targets.max(-1).indices

        targets_values = targets
        assert len(targets_values.shape) == 1

        for output in outputs:
            assert output.shape[0] == targets_values.shape[0]
            assert not torch.isnan(output).any(), "NaNs in outputs"
        # if unlabeled_outputs is not None:
        #     unlabeled_outputs = bootstrap_ensemble_outputs(unlabeled_outputs)

        # assert unlabeled_outputs is None, "unlabeled_outputs are always copied from outputs"
        # unlabeled_outputs = outputs

        repulsion_loss, modifier, reg_loss = None, None, None

        if self.weight > 0 and self.epoch + 1 > self.disagree_after_epoch:
            if self.disagree_below_threshold is not None:
                assert False, "disagree_below_threshold is not implemented"
                # assert self.modifier is None, \
                #     "Can't use modifier with disagree_below_threshold"
                # assert self.weight < 1, \
                #     "Can't have lambda == 1 with disagree_below_threshold"
                # assert self.use_always_labeled

                # masks = [
                #     (
                #             take_from_2d_tensor(
                #                 get_probs(output),
                #                 targets_values,
                #                 dim=-1
                #             )
                #         >
                #             self.disagree_below_threshold
                #     )
                #         for output
                #         in outputs
                # ]
                # # take samples which are low prob for all models
                # mask = torch.stack(masks).min(0).values
                # unlabeled_outputs = [
                #     output[~mask, ...]
                #         for output
                #         in outputs
                # ]
                # outputs = [output[mask, ...] for output in outputs]
                # targets = targets[mask, ...]
            if (
                    unlabeled_outputs is not None
                and
                    len(unlabeled_outputs) > 0
                and
                    len(unlabeled_outputs[0]) > 0
            ):
                repulsion_loss, modifier, unlabeled_outputs = get_repulsion_loss(
                    outputs,
                    unlabeled_outputs,
                    targets_values
                )

                reg_loss = get_regularizer(
                    self.reg_mode,
                    outputs,
                    unlabeled_outputs
                )

        if repulsion_loss is not None:
            assert not torch.isnan(repulsion_loss).any(), "NaNs in repulsion_loss"

        task_loss_value = torch.Tensor([0])[0]
        total_loss = task_loss_value.to(targets.device)
        if self.weight < 1:
            if len(outputs) > 0 and len(outputs[0]) > 0:
                task_loss_value = aggregate_tensors_by_func(
                    [self.task_loss(output, targets) for output in outputs]
                )

                total_loss = (1 - self.weight) * task_loss_value
        else:
            assert self.weight == 1
            assert self.disagree_after_epoch == 0, \
                "When lambda is 1 disagreement should start from the first epoch"

        if repulsion_loss is not None:

            repulsion_loss *= modifier

            if len(repulsion_loss.shape) > 0 and repulsion_loss.shape[0] > 1:
                repulsion_loss = repulsion_loss.mean()
            total_loss += self.manual_lambda * self.weight * repulsion_loss

        if reg_loss is not None:
            total_loss += self.reg_weight * self.weight * reg_loss

        loss_info = {
            "task_loss": task_loss_value.item(),
            "repulsion_loss": zero_if_none(repulsion_loss),
            "regularizer_loss": zero_if_none(reg_loss),
            "total_loss": total_loss.item()
        }

        # if self.log_this_batch:
        #     record_diversity(
        #         loss_info,
        #         outputs,
        #         torch.stack(outputs),
        #         metrics_mappings=metrics_mappings,
        #         name_prefix="ID_loss_"
        #     )
        #     if unlabeled_outputs is not None and not self.use_always_labeled:

        #         record_diversity(
        #             loss_info,
        #             unlabeled_outputs,
        #             torch.stack(unlabeled_outputs),
        #             metrics_mappings=metrics_mappings,
        #             name_prefix="OOD_loss_"
        #         )
        # gradients_info = {}
        # return total_loss, loss_info, gradients_info
        return total_loss


def budget_modifier(outputs, target):
    with torch.no_grad():
        ensemble_output = compute_ensemble_output(outputs)

        unreduced_ce = F.cross_entropy(
            ensemble_output,
            target,
            reduction='none'
        )
        divider = torch.pow(unreduced_ce.mean(0), 2)
        return unreduced_ce / divider


def get_regularizer(reg_mode, outputs, unlabeled_outputs):

    def chunk(outputs, heads):
        outputs_cat = torch.cat(
            outputs,
            axis=-1
        )
        chunked = torch.chunk(outputs_cat, heads, dim=-1)
        return chunked

    if reg_mode is None:
        return None

    assert reg_mode == "kl_backward"
    heads = len(outputs)

    yhat_chunked = chunk(outputs, heads)
    yhat_unlabeled_chunked = chunk(unlabeled_outputs, heads)

    preds = torch.stack(yhat_unlabeled_chunked).softmax(-1)

    avg_preds_source = (
        torch.stack(yhat_chunked).softmax(-1).mean([0, 1]).detach()
    )
    avg_preds_target = preds.mean(1)
    dist_source = Categorical(probs=avg_preds_source)
    dist_target = Categorical(probs=avg_preds_target)
    if reg_mode in ["kl_forward", "kl_ratio_f"]:
        kl = torch.distributions.kl.kl_divergence(dist_source, dist_target)
    elif reg_mode in ["kl_backward", "kl_ratio_b"]:
        kl = torch.distributions.kl.kl_divergence(dist_target, dist_source)
    reg_loss = kl.mean()
    return reg_loss


class A2DLoss(torch.nn.Module):
    def __init__(self, heads, dbat_loss_type='v1', reduction="mean"):
        super().__init__()
        self.heads = heads
        self.dbat_loss_type = dbat_loss_type
        self.reduction = reduction

    # input has shape [batch_size, heads * classes]
    def forward(self, logits):
        logits_chunked = torch.chunk(logits, self.heads, dim=-1)
        probs = torch.stack(logits_chunked, dim=0).softmax(-1)
        m_idx = torch.randint(0, self.heads, (1,)).item()
        # shape [models, batch, classes]
        return a2d_loss_impl(
            probs,
            m_idx,
            dbat_loss_type=self.dbat_loss_type,
            reduction=self.reduction
        )


def aggregate_tensors_by_func(input_list, func=torch.mean):
    return func(
        torch.stack(
            input_list
        )
    )


def a2d_loss_impl(probs, m_idx, dbat_loss_type='v1', reduction='mean'):

    if dbat_loss_type == 'v1':
        adv_loss = []

        p_1_s, indices = [], []

        for i, p_1 in enumerate(probs):
            if i == m_idx:
                continue
            p_1, idx = p_1.max(dim=1)
            p_1_s.append(p_1)
            indices.append(idx)

        p_2 = probs[m_idx]

        # probs for classes predicted by each other model
        p_2_s = [p_2[torch.arange(len(p_2)), max_idx] for max_idx in indices]

        for i in range(len(p_1_s)):
            al = (- torch.log(p_1_s[i] * (1-p_2_s[i]) + p_2_s[i] * (1-p_1_s[i]) + EPS))
            if reduction == 'mean':
                al = al.mean()
            else:
                assert reduction == 'none'

            adv_loss.append(al)

    else:
        raise NotImplementedError("v2 dbat is not implemented yet")

    if reduction == "none":
        agg_func = func_for_dim(torch.mean, 0)
    else:
        assert reduction == "mean"
        agg_func = torch.mean
    return aggregate_tensors_by_func(adv_loss, func=agg_func)


def func_for_dim(func, dim):

    def inner_func(*args, **kwargs):
        return func(*args, **kwargs, dim=dim)

    return inner_func


def compute_ensemble_output(
    outputs,
    weights=None,
    process_logits=None
):

    if process_logits is None:
        process_logits = lambda x: x

    if weights is None:
        weights = [1.0] * len(outputs)

    if stores_input(outputs):
        extractor = lambda x: x[1]
    else:
        extractor = lambda x: x

    return aggregate_tensors_by_func(
        [
            weight * process_logits(extractor(submodel_output).unsqueeze(0))
                for weight, submodel_output
                    in zip(weights, outputs)
        ],
        func=func_for_dim(torch.mean, dim=0)
    ).squeeze(0)


def stores_input(outputs):
    assert len(outputs) > 0
    output_0 = outputs[0]
    return isinstance(output_0, (list, tuple)) and len(output_0) == 2


def bootstrap_ensemble_outputs(outputs, assert_len=True):
    if_stores_input = stores_input(outputs)
    if assert_len:
        assert if_stores_input
    if if_stores_input:
        return [output[1] for output in outputs]
    else:
        return outputs
