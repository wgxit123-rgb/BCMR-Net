# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 2:14 下午
# @Author  : Haonan Wang
# @File    : Train_one_epoch.py
# @Software: PyCharm
import torch.optim
import os
import time
from utils import *
import UC_Config as config
import warnings
warnings.filterwarnings("ignore")

_AMP_SCALER = torch.cuda.amp.GradScaler(enabled=os.environ.get("UC_AMP", "1") == "1")


def _scalar(value):
    return value.item() if hasattr(value, "item") else value


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    # string += 'IoU:{:.3f} '.format(iou)
    # string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    # string += 'Acc:{:.3f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)
    # print summary


def _mask_4d(masks):
    return masks.float().unsqueeze(1) if masks.ndim == 3 else masks.float()


def _asymmetric_tversky_loss(preds, targets, alpha=0.3, beta=0.7, eps=1e-6):
    preds = preds.clamp(eps, 1.0 - eps)
    targets = targets.float()
    dims = (1, 2, 3)
    true_positive = (preds * targets).sum(dim=dims)
    false_positive = (preds * (1.0 - targets)).sum(dim=dims)
    false_negative = ((1.0 - preds) * targets).sum(dim=dims)
    score = (true_positive + eps) / (
        true_positive + alpha * false_positive + beta * false_negative + eps
    )
    return (1.0 - score).mean()


def _boundary_target(targets):
    dilated = torch.nn.functional.max_pool2d(targets, 3, stride=1, padding=1)
    eroded = -torch.nn.functional.max_pool2d(-targets, 3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def _boundary_dice_bce(preds, targets, eps=1e-6):
    preds = preds.clamp(eps, 1.0 - eps)
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    denominator = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + eps) / (denominator + eps))
    bce_loss = torch.nn.functional.binary_cross_entropy(preds, targets)
    return 0.5 * dice_loss.mean() + 0.5 * bce_loss


def _eibc_echo_randomize(images, masks):
    """Create a structure-preserving ultrasound echo-style view on the GPU."""
    masks = _mask_4d(masks).to(dtype=images.dtype)
    batch, _, height, width = images.shape
    permutation = torch.randperm(batch, device=images.device)

    image_float = images.float()
    low_frequency = torch.nn.functional.avg_pool2d(
        image_float, 31, stride=1, padding=15
    )
    high_frequency = image_float - low_frequency
    mix = torch.empty(batch, 1, 1, 1, device=images.device).uniform_(0.15, 0.45)
    source_mean = low_frequency.mean(dim=(2, 3), keepdim=True)
    source_std = low_frequency.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    donor_mean = source_mean[permutation]
    donor_std = source_std[permutation]
    normalized_low = (low_frequency - source_mean) / source_std
    transferred_low = normalized_low * donor_std + donor_mean
    mixed_low = (1.0 - mix) * low_frequency + mix * transferred_low
    echo_view = mixed_low + high_frequency

    # Ultrasound speckle strength differs inside and outside the lesion. A
    # smoothed mask avoids introducing a synthetic edge at the annotation.
    soft_mask = torch.nn.functional.avg_pool2d(masks, 9, stride=1, padding=4)
    sigma_fg = torch.empty(batch, 1, 1, 1, device=images.device).uniform_(0.02, 0.10)
    sigma_bg = torch.empty(batch, 1, 1, 1, device=images.device).uniform_(0.04, 0.16)
    sigma = sigma_bg * (1.0 - soft_mask) + sigma_fg * soft_mask
    fine_noise = torch.randn_like(echo_view)
    coarse_noise = torch.nn.functional.avg_pool2d(
        torch.randn_like(echo_view), 5, stride=1, padding=2
    )
    speckle = 1.0 + sigma * (0.75 * fine_noise + 0.25 * coarse_noise)
    styled = echo_view * speckle

    minimum = styled.amin(dim=(2, 3), keepdim=True)
    maximum = styled.amax(dim=(2, 3), keepdim=True)
    styled = (styled - minimum) / (maximum - minimum + 1e-6)
    gamma = torch.empty(batch, 1, 1, 1, device=images.device).uniform_(0.80, 1.25)
    return styled.clamp(0.0, 1.0).pow(gamma).to(dtype=images.dtype)


def _eibc_interval_loss(outputs, images, masks, criterion):
    if "core" not in outputs or "envelope" not in outputs:
        return outputs["main"].new_zeros(()), None
    target = _mask_4d(masks)
    core_target = -torch.nn.functional.max_pool2d(-target, 5, stride=1, padding=2)
    envelope_target = torch.nn.functional.max_pool2d(target, 5, stride=1, padding=2)
    auxiliary = 0.5 * (
        criterion(outputs["core"].float(), core_target)
        + criterion(outputs["envelope"].float(), envelope_target)
    )

    gray = images.float().mean(dim=1, keepdim=True)
    sobel_x = gray.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    grad_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
    gradient = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
    gradient = gradient / (gradient.amax(dim=(2, 3), keepdim=True) + 1e-6)
    band = (envelope_target - core_target).clamp(0.0, 1.0)
    ambiguity_target = band * (0.25 + 0.75 * (1.0 - gradient).clamp(0.0, 1.0))
    interval_width = (
        torch.sigmoid(outputs["envelope"].float())
        - torch.sigmoid(outputs["core"].float())
    ).clamp(0.0, 1.0)
    width_loss = torch.nn.functional.smooth_l1_loss(
        interval_width, ambiguity_target
    )
    return auxiliary + 0.5 * width_loss, interval_width.detach()


def _dcr_geometry_loss(outputs, sdf_targets):
    """Supervise continuous geometry and align it only near uncertain margins."""
    if sdf_targets is None or "sdf" not in outputs:
        return outputs["main"].new_zeros(())

    signed_distance = outputs["sdf"].float()
    target = torch.nn.functional.interpolate(
        sdf_targets.float(), size=signed_distance.shape[-2:],
        mode="bilinear", align_corners=False,
    )
    boundary_weight = 1.0 + 2.0 * torch.exp(-target.abs() / 0.15)
    distance_loss = (
        torch.nn.functional.smooth_l1_loss(
            signed_distance, target, reduction="none"
        ) * boundary_weight
    ).mean()

    if "distance_logits" not in outputs:
        return distance_loss
    if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1":
        main_probability = torch.sigmoid(outputs["main"].float())
    else:
        main_probability = outputs["main"].float().clamp(0.0, 1.0)
    distance_probability = torch.sigmoid(outputs["distance_logits"].float())
    boundary_band = torch.exp(-target.abs() / 0.20)
    alignment = (
        boundary_band * (main_probability - distance_probability).square()
    ).sum() / (boundary_band.sum() + 1e-6)
    alignment_weight = float(os.environ.get("UC_DCR_ALIGNMENT_WEIGHT", "0.03"))
    return distance_loss + alignment_weight * alignment


def _psca_probability(preds):
    if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1":
        return torch.sigmoid(preds.float())
    return preds.float().clamp(0.0, 1.0)


def _psca_scale_risk_costs(mask):
    """Allocate recover/suppress costs from lesion scale during training."""
    area_reference = float(
        os.environ.get("UC_PSCA_AREA_REFERENCE", "0.06")
    )
    minimum = float(
        os.environ.get("UC_PSCA_SCALE_COST_MIN", "0.5")
    )
    maximum = float(
        os.environ.get("UC_PSCA_SCALE_COST_MAX", "2.0")
    )
    area = mask.float().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
    relative_scale = torch.sqrt(
        area / max(area_reference, 1e-4)
    ).clamp(minimum, maximum)
    preserve_cost = torch.ones_like(relative_scale)
    recover_cost = relative_scale
    suppress_cost = relative_scale.reciprocal()
    return torch.cat(
        [preserve_cost, recover_cost, suppress_cost],
        dim=1,
    )


def _psca_scale_equitable_weights(mask):
    """Give small lesions a bounded share of the per-image overlap gradient."""
    reference = float(
        os.environ.get("UC_PSCA_IOU_AREA_REFERENCE", "0.06")
    )
    minimum = float(
        os.environ.get("UC_PSCA_IOU_WEIGHT_MIN", "0.5")
    )
    maximum = float(
        os.environ.get("UC_PSCA_IOU_WEIGHT_MAX", "2.0")
    )
    area = mask.float().mean(dim=(1, 2, 3)).clamp_min(1e-4)
    weights = torch.sqrt(
        max(reference, 1e-4) / area
    ).clamp(minimum, maximum)
    return weights / weights.mean().clamp_min(1e-6)


def _psca_scale_equitable_iou_loss(preds, targets, eps=1e-6):
    """Optimize mean per-image IoU without letting large lesions dominate."""
    probability = _psca_probability(preds)
    target = _mask_4d(targets)
    dims = (1, 2, 3)
    intersection = (probability * target).sum(dim=dims)
    union = (
        probability + target - probability * target
    ).sum(dim=dims)
    soft_iou = (intersection + eps) / (union + eps)
    weights = _psca_scale_equitable_weights(target)
    return (weights * (1.0 - soft_iou)).mean()


def _psca_merge_projected_gradients(
    primary_gradients,
    auxiliary_gradients,
    max_aux_ratio=0.5,
    eps=1e-12,
):
    """Project conflicting arbitration gradients away from region gradients."""
    paired = [
        (primary, auxiliary)
        for primary, auxiliary in zip(
            primary_gradients,
            auxiliary_gradients,
        )
        if primary is not None and auxiliary is not None
    ]
    if not paired:
        return list(primary_gradients), {
            "conflict": False,
            "cosine": 0.0,
            "aux_scale": 0.0,
        }

    primary_norm_sq = sum(
        gradient.detach().square().sum()
        for gradient, _ in paired
    )
    auxiliary_norm_sq = sum(
        gradient.detach().square().sum()
        for _, gradient in paired
    )
    dot = sum(
        primary.detach().mul(auxiliary.detach()).sum()
        for primary, auxiliary in paired
    )
    cosine = dot / (
        primary_norm_sq.sqrt()
        * auxiliary_norm_sq.sqrt()
        + eps
    )
    conflict = bool(dot.item() < 0.0)
    coefficient = (
        dot / (primary_norm_sq + eps)
        if conflict
        else dot.new_zeros(())
    )

    projected_auxiliary = []
    for primary, auxiliary in zip(
        primary_gradients,
        auxiliary_gradients,
    ):
        if auxiliary is None:
            projected_auxiliary.append(None)
        elif primary is None or not conflict:
            projected_auxiliary.append(auxiliary)
        else:
            projected_auxiliary.append(
                auxiliary - coefficient * primary
            )

    projected_norm_sq = sum(
        gradient.detach().square().sum()
        for gradient in projected_auxiliary
        if gradient is not None
    )
    if float(projected_norm_sq) == 0.0:
        auxiliary_scale = dot.new_zeros(())
    else:
        auxiliary_scale = torch.clamp(
            float(max_aux_ratio)
            * primary_norm_sq.sqrt()
            / (projected_norm_sq.sqrt() + eps),
            max=1.0,
        )

    merged = []
    for primary, auxiliary in zip(
        primary_gradients,
        projected_auxiliary,
    ):
        if primary is None and auxiliary is None:
            merged.append(None)
        elif primary is None:
            merged.append(auxiliary_scale * auxiliary)
        elif auxiliary is None:
            merged.append(primary)
        else:
            merged.append(primary + auxiliary_scale * auxiliary)

    return merged, {
        "conflict": conflict,
        "cosine": float(cosine.detach()),
        "aux_scale": float(auxiliary_scale.detach()),
    }


def _psca_region_safe_step(
    primary_loss,
    auxiliary_loss,
    model,
    optimizer,
):
    """Apply one-sided gradient projection while preserving the region task."""
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer.zero_grad()
    primary_gradients = torch.autograd.grad(
        primary_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    auxiliary_gradients = torch.autograd.grad(
        auxiliary_loss,
        parameters,
        allow_unused=True,
    )
    max_aux_ratio = float(
        os.environ.get("UC_PSCA_MAX_AUX_GRAD_RATIO", "0.5")
    )
    merged, diagnostics = _psca_merge_projected_gradients(
        primary_gradients,
        auxiliary_gradients,
        max_aux_ratio=max_aux_ratio,
    )
    for parameter, gradient in zip(parameters, merged):
        parameter.grad = gradient
    optimizer.step()
    return diagnostics


def _psca_policy_loss(outputs, masks, epoch, mode):
    """Teach PSCA's existing router explicit preserve/recover/suppress roles."""
    states_by_scale = outputs.get("psca_states", ())
    if not states_by_scale:
        return outputs["main"].new_zeros(()), {}

    mask = _mask_4d(masks)
    probability = _psca_probability(outputs["main"]).detach()
    scale_risk_costs = (
        _psca_scale_risk_costs(mask)
        if mode == "risk_projected"
        else None
    )
    scale_losses = []
    state_mass = []
    for states in states_by_scale:
        target = torch.nn.functional.interpolate(
            mask,
            size=states.shape[-2:],
            mode="nearest",
        )
        prediction = torch.nn.functional.interpolate(
            probability,
            size=states.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        recover = target * (1.0 - prediction)
        suppress = (1.0 - target) * prediction
        preserve = (1.0 - recover - suppress).clamp(0.0, 1.0)
        policy_target = torch.cat(
            [preserve, recover, suppress],
            dim=1,
        ).detach()

        class_mass = policy_target.mean(dim=(0, 2, 3)).clamp_min(1e-3)
        class_weight = (
            class_mass.mean() / class_mass
        ).sqrt().clamp(0.5, 3.0)
        class_weight = class_weight / class_weight.mean()

        log_states = states.float().clamp_min(1e-6).log()
        weighted_target = (
            policy_target
            * class_weight.view(1, 3, 1, 1)
        )
        if scale_risk_costs is not None:
            weighted_target = weighted_target * scale_risk_costs
            target_normalizer = (
                policy_target * scale_risk_costs
            ).sum(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            target_normalizer = 1.0
        pixel_loss = -(
            weighted_target * log_states
        ).sum(dim=1, keepdim=True) / target_normalizer

        error_strength = (recover + suppress).detach()
        pixel_weight = 0.35 + 0.65 * error_strength
        if mode in {
            "boundary",
            "counterfactual",
            "risk_projected",
            "region_projected",
            "overlap_guard",
        }:
            boundary = _boundary_target(target)
            boundary_band = torch.nn.functional.max_pool2d(
                boundary,
                5,
                stride=1,
                padding=2,
            )
            uncertainty = (
                4.0 * prediction * (1.0 - prediction)
            ).detach()
            pixel_weight = pixel_weight * (
                1.0
                + 1.5
                * boundary_band
                * (0.5 + 0.5 * uncertainty)
            )

        scale_losses.append(
            (pixel_loss * pixel_weight).sum()
            / (pixel_weight.sum() + 1e-6)
        )
        state_mass.append(states.float().mean(dim=(0, 2, 3)).detach())

    warmup = int(os.environ.get("UC_PSCA_POLICY_WARMUP_EPOCHS", "5"))
    ramp_epochs = max(
        1,
        int(os.environ.get("UC_PSCA_POLICY_RAMP_EPOCHS", "20")),
    )
    ramp = max(0.0, min(1.0, (epoch + 1 - warmup) / ramp_epochs))
    policy_weight = float(
        os.environ.get("UC_PSCA_POLICY_WEIGHT", "0.10")
    )
    loss = (
        policy_weight
        * ramp
        * sum(scale_losses)
        / len(scale_losses)
    )
    diagnostics = {
        "ramp": ramp,
        "state_mass": torch.stack(state_mass).mean(dim=0),
    }
    return loss, diagnostics


def _psca_soft_boundary_loss(preds, masks):
    probability = _psca_probability(preds)
    target = _mask_4d(masks)
    predicted_boundary = _boundary_target(probability)
    target_boundary = _boundary_target(target)
    return _boundary_dice_bce(predicted_boundary, target_boundary)


def _psca_counterfactual_view(images, masks):
    """Intervene on lesion margins without changing the segmentation target."""
    target = _mask_4d(masks).to(dtype=images.dtype)
    boundary = _boundary_target(target)
    band = torch.nn.functional.max_pool2d(
        boundary,
        7,
        stride=1,
        padding=3,
    )
    inner_band = band * target
    outer_band = band * (1.0 - target)

    blurred = torch.nn.functional.avg_pool2d(
        images.float(),
        9,
        stride=1,
        padding=4,
    ).to(dtype=images.dtype)
    speckle = torch.randn_like(images)
    speckle = 0.7 * speckle + 0.3 * torch.nn.functional.avg_pool2d(
        speckle,
        5,
        stride=1,
        padding=2,
    )

    batch = images.shape[0]
    recover_case = (
        torch.arange(batch, device=images.device) % 2 == 0
    ).to(images.dtype).view(batch, 1, 1, 1)
    suppress_case = 1.0 - recover_case
    recover_region = recover_case * inner_band
    suppress_region = suppress_case * outer_band

    recover_strength = torch.empty(
        batch,
        1,
        1,
        1,
        device=images.device,
        dtype=images.dtype,
    ).uniform_(0.35, 0.65)
    suppress_strength = torch.empty(
        batch,
        1,
        1,
        1,
        device=images.device,
        dtype=images.dtype,
    ).uniform_(0.08, 0.18)

    styled = (
        images * (1.0 - recover_strength * recover_region)
        + blurred * recover_strength * recover_region
    )
    styled = styled * (
        1.0 + suppress_strength * suppress_region * speckle
    )
    minimum = styled.amin(dim=(2, 3), keepdim=True)
    maximum = styled.amax(dim=(2, 3), keepdim=True)
    styled = (styled - minimum) / (maximum - minimum + 1e-6)
    diagnostics = {
        "recover_region": recover_region,
        "suppress_region": suppress_region,
    }
    return styled.to(dtype=images.dtype), diagnostics


def _psca_counterfactual_state_loss(
    anchor_outputs,
    counterfactual_outputs,
    intervention,
):
    anchor_states = anchor_outputs.get("psca_states", ())
    altered_states = counterfactual_outputs.get("psca_states", ())
    if not anchor_states or not altered_states:
        return counterfactual_outputs["main"].new_zeros(())

    margin = float(os.environ.get("UC_PSCA_CF_STATE_MARGIN", "0.02"))
    losses = []
    for anchor, altered in zip(anchor_states, altered_states):
        recover_region = torch.nn.functional.interpolate(
            intervention["recover_region"],
            size=altered.shape[-2:],
            mode="nearest",
        )
        suppress_region = torch.nn.functional.interpolate(
            intervention["suppress_region"],
            size=altered.shape[-2:],
            mode="nearest",
        )
        active_region = (recover_region + suppress_region).clamp(0.0, 1.0)
        preserve_region = 1.0 - active_region

        recover_response = torch.relu(
            margin
            - (
                altered[:, 1:2]
                - anchor[:, 1:2].detach()
            )
        )
        suppress_response = torch.relu(
            margin
            - (
                altered[:, 2:3]
                - anchor[:, 2:3].detach()
            )
        )
        response_loss = (
            (recover_response * recover_region).sum()
            / (recover_region.sum() + 1e-6)
            + (suppress_response * suppress_region).sum()
            / (suppress_region.sum() + 1e-6)
        )
        stable_loss = (
            (
                altered[:, 0:1]
                - anchor[:, 0:1].detach()
            ).abs()
            * preserve_region
        ).sum() / (preserve_region.sum() + 1e-6)
        losses.append(response_loss + 0.25 * stable_loss)
    return sum(losses) / len(losses)


def _psca_lateral_equivariance_loss(
    anchor_outputs,
    reflected_outputs,
    masks,
):
    """Align predictions and arbitration states under lateral reflection."""
    anchor_probability = _psca_probability(
        anchor_outputs["main"]
    ).detach()
    reflected_probability = torch.flip(
        _psca_probability(reflected_outputs["main"]),
        dims=(-1,),
    )
    target = _mask_4d(masks)
    boundary = torch.nn.functional.max_pool2d(
        _boundary_target(target),
        7,
        stride=1,
        padding=3,
    )
    anchor_confidence = (
        2.0 * (anchor_probability - 0.5).abs()
    ).detach()
    pixel_weight = (
        0.25
        + 0.75
        * anchor_confidence
        * (0.5 + 0.5 * boundary)
    )
    prediction_loss = (
        pixel_weight
        * (reflected_probability - anchor_probability).square()
    ).sum() / (pixel_weight.sum() + 1e-6)

    anchor_states = anchor_outputs.get("psca_states", ())
    reflected_states = reflected_outputs.get("psca_states", ())
    state_losses = []
    for anchor, reflected in zip(anchor_states, reflected_states):
        anchor = anchor.detach()
        reflected = torch.flip(reflected, dims=(-1,))
        state_confidence = anchor.amax(dim=1, keepdim=True).detach()
        scale_boundary = torch.nn.functional.interpolate(
            boundary,
            size=anchor.shape[-2:],
            mode="nearest",
        )
        state_weight = (
            0.25
            + 0.75
            * state_confidence
            * (0.5 + 0.5 * scale_boundary)
        )
        state_losses.append(
            (
                state_weight
                * (reflected - anchor).square().sum(
                    dim=1,
                    keepdim=True,
                )
            ).sum()
            / (state_weight.sum() + 1e-6)
        )
    state_loss = (
        sum(state_losses) / len(state_losses)
        if state_losses
        else prediction_loss.new_zeros(())
    )
    return prediction_loss, state_loss


##################################################################################
#=================================================================================
#          Train One Epoch
#=================================================================================
##################################################################################
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger,loss_min,iou_max,dice_max):
    logging_mode = 'Train' if model.training else 'Val'
    metrics_interval = int(os.environ.get("UC_TRAIN_METRICS_INTERVAL", "1"))
    metrics_enabled = metrics_interval > 0

    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0

    dices = []
    projection_steps = 0
    projection_conflicts = 0
    projection_cosine_sum = 0.0
    projection_scale_sum = 0.0
    for i, (sampled_batch, names) in enumerate(loader, 1):

        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # Take variable and put them to GPU
        images, masks = sampled_batch['image'], sampled_batch['label']
        sdf_targets = sampled_batch.get('sdf')
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)
        if sdf_targets is not None:
            sdf_targets = sdf_targets.cuda(non_blocking=True).float().unsqueeze(1)
        if os.environ.get("UC_CHANNELS_LAST", "1") == "1":
            images = images.contiguous(memory_format=torch.channels_last)


        # ====================================================
        #             Compute loss
        # ====================================================

        use_amp = os.environ.get("UC_AMP", "1") == "1"
        raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        use_echo_view = bool(
            model.training and getattr(raw_model, "use_echo_randomization", False)
        )
        psca_policy_mode = getattr(raw_model, "policy_mode", None)
        use_psca_counterfactual = bool(
            model.training and psca_policy_mode == "counterfactual"
        )
        use_psca_region_projection = bool(
            model.training
            and psca_policy_mode in {
                "risk_projected",
                "region_projected",
            }
        )
        use_psca_lateral_equivariance = bool(
            model.training
            and psca_policy_mode == "axial_equivariant"
        )
        eibc_outputs = None
        with torch.cuda.amp.autocast(enabled=use_amp):
            model_outputs = model(images)
            if isinstance(model_outputs, dict):
                eibc_outputs = model_outputs
                preds = model_outputs["main"]
                sdf_preds = ()
                coarse_preds = ()
                route_losses = ()
                boundary_preds = ()
            elif isinstance(model_outputs, (tuple, list)):
                preds = model_outputs[0]
                sdf_preds = model_outputs[1] if len(model_outputs) > 1 else ()
                coarse_preds = model_outputs[2] if len(model_outputs) > 2 else ()
                route_losses = model_outputs[3] if len(model_outputs) > 3 else ()
                boundary_preds = model_outputs[4] if len(model_outputs) > 4 else ()
            else:
                preds = model_outputs
                sdf_preds = ()
                coarse_preds = ()
                route_losses = ()
                boundary_preds = ()
        with torch.cuda.amp.autocast(enabled=False):
            out_loss = criterion(preds.float(), masks.float())  # Loss
            psca_projected_auxiliary = None
            if eibc_outputs is not None:
                interval_weight = float(os.environ.get("UC_EIBC_INTERVAL_WEIGHT", "0.20"))
                consistency_weight = float(os.environ.get("UC_EIBC_CONSISTENCY_WEIGHT", "0.10"))
                interval_loss, interval_width = _eibc_interval_loss(
                    eibc_outputs, images, masks, criterion
                )
                out_loss = out_loss + interval_weight * interval_loss
                if "sdf" in eibc_outputs:
                    dcr_weight = float(os.environ.get("UC_DCR_DISTANCE_WEIGHT", "0.15"))
                    out_loss = out_loss + dcr_weight * _dcr_geometry_loss(
                        eibc_outputs, sdf_targets
                    )
                if (
                    "psca_states" in eibc_outputs
                    and not use_psca_lateral_equivariance
                ):
                    policy_loss, _ = _psca_policy_loss(
                        eibc_outputs,
                        masks,
                        epoch,
                        psca_policy_mode,
                    )
                    if use_psca_region_projection:
                        psca_projected_auxiliary = policy_loss
                    else:
                        out_loss = out_loss + policy_loss
                    if psca_policy_mode in {
                        "boundary",
                        "counterfactual",
                        "risk_projected",
                        "region_projected",
                        "overlap_guard",
                    }:
                        boundary_weight = float(
                            os.environ.get(
                                "UC_PSCA_BOUNDARY_WEIGHT",
                                "0.04",
                            )
                        )
                        warmup = int(
                            os.environ.get(
                                "UC_PSCA_POLICY_WARMUP_EPOCHS",
                                "5",
                            )
                        )
                        ramp_epochs = max(
                            1,
                            int(
                                os.environ.get(
                                    "UC_PSCA_POLICY_RAMP_EPOCHS",
                                    "20",
                                )
                            ),
                        )
                        boundary_ramp = max(
                            0.0,
                            min(
                                1.0,
                                (epoch + 1 - warmup) / ramp_epochs,
                            ),
                        )
                        boundary_loss = (
                            boundary_weight
                            * boundary_ramp
                            * _psca_soft_boundary_loss(preds, masks)
                        )
                        if use_psca_region_projection:
                            psca_projected_auxiliary = (
                                psca_projected_auxiliary
                                + boundary_loss
                            )
                        else:
                            out_loss = out_loss + boundary_loss
                        if psca_policy_mode == "overlap_guard":
                            overlap_weight = float(
                                os.environ.get(
                                    "UC_PSCA_SCALE_IOU_WEIGHT",
                                    "0.05",
                                )
                            )
                            out_loss = out_loss + (
                                overlap_weight
                                * boundary_ramp
                                * _psca_scale_equitable_iou_loss(
                                    preds,
                                    masks,
                                )
                            )
            if sdf_targets is not None and len(sdf_preds) > 0:
                sdf_weight = float(os.environ.get("UC_SDF_LOSS_WEIGHT", "0.15"))
                coarse_weight = float(os.environ.get("UC_COARSE_LOSS_WEIGHT", "0.10"))
                sdf_loss = 0.0
                coarse_loss = 0.0
                for sdf_pred in sdf_preds:
                    target = torch.nn.functional.interpolate(
                        sdf_targets, size=sdf_pred.shape[-2:], mode='bilinear', align_corners=False
                    )
                    boundary_weight = 1.0 + 2.0 * torch.exp(-target.abs() / 0.15)
                    sdf_loss = sdf_loss + (
                        torch.nn.functional.smooth_l1_loss(
                            sdf_pred.float(), target, reduction='none'
                        ) * boundary_weight
                    ).mean()
                for coarse_pred in coarse_preds:
                    mask_target = masks.float().unsqueeze(1) if masks.ndim == 3 else masks.float()
                    mask_target = torch.nn.functional.interpolate(
                        mask_target, size=coarse_pred.shape[-2:], mode='nearest'
                    )
                    coarse_loss = coarse_loss + criterion(coarse_pred.float(), mask_target)
                out_loss = out_loss + sdf_weight * sdf_loss / len(sdf_preds)
                if len(coarse_preds) > 0:
                    out_loss = out_loss + coarse_weight * coarse_loss / len(coarse_preds)
            route_warmup = int(os.environ.get("UC_ROUTE_WARMUP_EPOCHS", "20"))
            if epoch < route_warmup and len(route_losses) > 0:
                route_weight = float(os.environ.get("UC_ROUTE_LOSS_WEIGHT", "0.01"))
                route_loss = sum(route.float().mean() for route in route_losses) / len(route_losses)
                out_loss = out_loss + route_weight * route_loss
            if len(boundary_preds) > 0:
                mask_target = _mask_4d(masks)
                tversky_weight = float(os.environ.get("UC_TVERSKY_LOSS_WEIGHT", "0.15"))
                boundary_weight = float(os.environ.get("UC_BOUNDARY_LOSS_WEIGHT", "0.10"))
                out_loss = out_loss + tversky_weight * _asymmetric_tversky_loss(
                    preds.float(), mask_target
                )
                boundary_loss = 0.0
                for boundary_pred in boundary_preds:
                    resized_target = torch.nn.functional.interpolate(
                        mask_target, size=boundary_pred.shape[-2:], mode='nearest'
                    )
                    boundary_loss = boundary_loss + _boundary_dice_bce(
                        boundary_pred.float(), _boundary_target(resized_target)
                    )
                out_loss = out_loss + boundary_weight * boundary_loss / len(boundary_preds)


        eibc_manual_backward = False
        if use_psca_region_projection:
            if use_amp:
                raise RuntimeError(
                    "PSCA region projection requires UC_AMP=0"
                )
            if psca_projected_auxiliary is None:
                raise RuntimeError(
                    "PSCA region projection auxiliary loss is missing"
                )
            diagnostics = _psca_region_safe_step(
                out_loss,
                psca_projected_auxiliary,
                model,
                optimizer,
            )
            projection_steps += 1
            projection_conflicts += int(diagnostics["conflict"])
            projection_cosine_sum += diagnostics["cosine"]
            projection_scale_sum += diagnostics["aux_scale"]
            out_loss = (
                out_loss.detach()
                + psca_projected_auxiliary.detach()
            )
            eibc_manual_backward = True

        if model.training and eibc_outputs is not None and use_echo_view:
            # Backpropagate the two views sequentially. This keeps the effective
            # batch size at 32 without retaining two BCMR computation graphs.
            optimizer.zero_grad()
            anchor_probability = torch.sigmoid(preds.float()).detach()
            anchor_width = interval_width
            original_loss = 0.5 * out_loss
            if use_amp:
                _AMP_SCALER.scale(original_loss).backward()
            else:
                original_loss.backward()

            styled_images = _eibc_echo_randomize(images, masks)
            with torch.cuda.amp.autocast(enabled=use_amp):
                styled_outputs = model(styled_images)
            with torch.cuda.amp.autocast(enabled=False):
                styled_main = styled_outputs["main"].float()
                styled_interval_loss, styled_width = _eibc_interval_loss(
                    styled_outputs, styled_images, masks, criterion
                )
                styled_loss = 0.5 * criterion(styled_main, masks.float())
                styled_loss = styled_loss + 0.5 * interval_weight * styled_interval_loss
                styled_probability = torch.sigmoid(styled_main)
                if anchor_width is None:
                    confidence = torch.ones_like(anchor_probability)
                else:
                    mean_width = anchor_width
                    if styled_width is not None:
                        mean_width = 0.5 * (mean_width + styled_width)
                    confidence = 0.25 + 0.75 * (1.0 - mean_width).clamp(0.0, 1.0)
                consistency = (
                    confidence * (anchor_probability - styled_probability).square()
                ).mean()
                styled_loss = styled_loss + consistency_weight * consistency
            if use_amp:
                _AMP_SCALER.scale(styled_loss).backward()
                _AMP_SCALER.step(optimizer)
                _AMP_SCALER.update()
            else:
                styled_loss.backward()
                optimizer.step()
            out_loss = original_loss.detach() + styled_loss.detach()
            eibc_manual_backward = True

        if use_psca_counterfactual:
            optimizer.zero_grad()
            anchor_probability = _psca_probability(preds).detach()
            anchor_outputs = eibc_outputs
            original_loss = 0.5 * out_loss
            if use_amp:
                _AMP_SCALER.scale(original_loss).backward()
            else:
                original_loss.backward()

            counterfactual_images, intervention = _psca_counterfactual_view(
                images,
                masks,
            )
            with torch.cuda.amp.autocast(enabled=use_amp):
                counterfactual_outputs = model(counterfactual_images)
                counterfactual_preds = counterfactual_outputs["main"]
            with torch.cuda.amp.autocast(enabled=False):
                counterfactual_loss = 0.5 * criterion(
                    counterfactual_preds.float(),
                    masks.float(),
                )
                policy_loss, _ = _psca_policy_loss(
                    counterfactual_outputs,
                    masks,
                    epoch,
                    "counterfactual",
                )
                counterfactual_loss = (
                    counterfactual_loss + 0.5 * policy_loss
                )
                boundary_weight = float(
                    os.environ.get(
                        "UC_PSCA_BOUNDARY_WEIGHT",
                        "0.04",
                    )
                )
                counterfactual_loss = counterfactual_loss + (
                    0.5
                    * boundary_weight
                    * _psca_soft_boundary_loss(
                        counterfactual_preds,
                        masks,
                    )
                )
                consistency_weight = float(
                    os.environ.get(
                        "UC_PSCA_CF_CONSISTENCY_WEIGHT",
                        "0.05",
                    )
                )
                state_weight = float(
                    os.environ.get(
                        "UC_PSCA_CF_STATE_WEIGHT",
                        "0.03",
                    )
                )
                counterfactual_probability = _psca_probability(
                    counterfactual_preds
                )
                boundary = torch.nn.functional.max_pool2d(
                    _boundary_target(_mask_4d(masks)),
                    7,
                    stride=1,
                    padding=3,
                )
                consistency = (
                    (1.0 + boundary)
                    * (
                        counterfactual_probability
                        - anchor_probability
                    ).square()
                ).mean()
                state_response = _psca_counterfactual_state_loss(
                    anchor_outputs,
                    counterfactual_outputs,
                    intervention,
                )
                counterfactual_loss = counterfactual_loss + (
                    consistency_weight * consistency
                    + state_weight * state_response
                )
            if use_amp:
                _AMP_SCALER.scale(counterfactual_loss).backward()
                _AMP_SCALER.step(optimizer)
                _AMP_SCALER.update()
            else:
                counterfactual_loss.backward()
                optimizer.step()
            out_loss = (
                original_loss.detach()
                + counterfactual_loss.detach()
            )
            eibc_manual_backward = True

        if use_psca_lateral_equivariance:
            optimizer.zero_grad()
            anchor_outputs = {
                "main": preds.detach(),
                "psca_states": tuple(
                    states.detach()
                    for states in eibc_outputs["psca_states"]
                ),
            }
            original_loss = 0.5 * out_loss
            if use_amp:
                _AMP_SCALER.scale(original_loss).backward()
            else:
                original_loss.backward()

            reflected_images = torch.flip(images, dims=(-1,))
            reflected_masks = torch.flip(masks, dims=(-1,))
            with torch.cuda.amp.autocast(enabled=use_amp):
                reflected_outputs = model(reflected_images)
                reflected_preds = reflected_outputs["main"]
            with torch.cuda.amp.autocast(enabled=False):
                reflected_loss = 0.5 * criterion(
                    reflected_preds.float(),
                    reflected_masks.float(),
                )
                prediction_consistency, state_consistency = (
                    _psca_lateral_equivariance_loss(
                        anchor_outputs,
                        reflected_outputs,
                        masks,
                    )
                )
                warmup = int(
                    os.environ.get(
                        "UC_PSCA_EQUIV_WARMUP_EPOCHS",
                        "5",
                    )
                )
                ramp_epochs = max(
                    1,
                    int(
                        os.environ.get(
                            "UC_PSCA_EQUIV_RAMP_EPOCHS",
                            "20",
                        )
                    ),
                )
                ramp = max(
                    0.0,
                    min(
                        1.0,
                        (epoch + 1 - warmup) / ramp_epochs,
                    ),
                )
                prediction_weight = float(
                    os.environ.get(
                        "UC_PSCA_EQUIV_PRED_WEIGHT",
                        "0.05",
                    )
                )
                state_weight = float(
                    os.environ.get(
                        "UC_PSCA_EQUIV_STATE_WEIGHT",
                        "0.02",
                    )
                )
                reflected_loss = reflected_loss + ramp * (
                    prediction_weight * prediction_consistency
                    + state_weight * state_consistency
                )
            if use_amp:
                _AMP_SCALER.scale(reflected_loss).backward()
                _AMP_SCALER.step(optimizer)
                _AMP_SCALER.update()
            else:
                reflected_loss.backward()
                optimizer.step()
            out_loss = (
                original_loss.detach()
                + reflected_loss.detach()
            )
            eibc_manual_backward = True

        if model.training and not eibc_manual_backward:
            optimizer.zero_grad()
            if use_amp:
                _AMP_SCALER.scale(out_loss).backward()
                _AMP_SCALER.step(optimizer)
                _AMP_SCALER.update()
            else:
                out_loss.backward()
                optimizer.step()

        # print(masks.size())
        # print(preds.size())


        train_iou = 0.0
        train_dice = 0.0
        should_compute_metrics = logging_mode == 'Val' or (
            metrics_enabled and (
                i % metrics_interval == 0 or i == len(loader)
            )
        )
        if should_compute_metrics:
            with torch.no_grad():
                train_iou = iou_on_batch(masks, preds)
                train_dice = criterion._show_dice(preds.detach().clone(), masks.float().detach().clone())

        batch_time = time.time() - end
        # train_acc = acc_on_batch(masks,preds)
        if config.vis_frequency > 0 and epoch % config.vis_frequency == 0 and logging_mode == 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        if should_compute_metrics:
            dices.append(train_dice)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss.detach()
        iou_sum += len(images) * train_iou
        # acc_sum += len(images) * train_acc
        dice_sum += len(images) * train_dice

        if i == len(loader):
            average_loss = loss_sum / (config.batch_size*(i-1) + len(images))
            average_time = time_sum / (config.batch_size*(i-1) + len(images))
            train_iou_average = iou_sum / (config.batch_size*(i-1) + len(images))
            # train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images))
            if projection_steps > 0:
                logger.info(
                    "PSCA projected optimization conflict_rate={:.4f} "
                    "mean_cosine={:.4f} mean_aux_scale={:.4f}".format(
                        projection_conflicts / projection_steps,
                        projection_cosine_sum / projection_steps,
                        projection_scale_sum / projection_steps,
                    )
                )
        else:
            average_loss = loss_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            train_iou_average = iou_sum / (i * config.batch_size)
            # train_acc_average = acc_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size)

        end = time.time()
        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss, loss_name, batch_time,
                          average_loss, average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, 0, 0,  logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)

        #if config.tensorboard:
            #step = epoch * len(loader) + i
            #writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            # plot metrics in tensorboard
            #writer.add_scalar(logging_mode + '_iou', train_iou, step)
            # writer.add_scalar(logging_mode + '_acc', train_acc, step)
            #writer.add_scalar(logging_mode + '_dice', train_dice, step)

    step1 = epoch+1
    if writer is None:
        if lr_scheduler is not None:
            lr_scheduler.step()
        return average_loss, train_dice_avg, loss_min, iou_max, dice_max

    writer.add_scalar(logging_mode +'_'+loss_name + '_avg', average_loss.item(), step1)
    if average_loss<loss_min:
        loss_min=average_loss
    writer.add_scalar(logging_mode +'_'+ loss_name + '_min', _scalar(loss_min), step1)


    # plot metrics in tensorboard
    writer.add_scalar(logging_mode + '_iou_avg', train_iou_average.item(), step1)
    if train_iou_average>iou_max:
        iou_max=train_iou_average
    writer.add_scalar(logging_mode + '_iou_max', _scalar(iou_max), step1)
    # writer.add_scalar(logging_mode + '_acc', train_acc, step)


    writer.add_scalar(logging_mode + '_dice_avg', train_dice_avg.item(), step1)
    if train_dice_avg>dice_max:
        dice_max=train_dice_avg
    writer.add_scalar(logging_mode + '_dice_max', _scalar(dice_max), step1)

    if lr_scheduler is not None:
        lr_scheduler.step()
    # if epoch + 1 > 10: # Plateau
    #     if lr_scheduler is not None:
    #         lr_scheduler.step(train_dice_avg)

    return average_loss, train_dice_avg,loss_min,iou_max,dice_max
