import torch
import torch.nn.functional as F


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """SupCon loss (Khosla et al., 2020). Pulls same-class embeddings together, pushes others apart."""
    embeddings = F.normalize(embeddings, p=2, dim=1)
    batch_size = embeddings.shape[0]

    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.T).float()
    positive_mask.fill_diagonal_(0)
    negative_mask = 1 - (labels == labels.T).float()

    # Subtract max for numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    exp_logits = torch.exp(logits)
    # Denominator: all samples except self
    denom = exp_logits * (1 - torch.eye(batch_size, device=embeddings.device))
    log_prob = logits - torch.log(denom.sum(dim=1, keepdim=True) + 1e-8)

    num_positives = positive_mask.sum(dim=1)
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)

    loss = -mean_log_prob_pos
    loss = loss[num_positives > 0].mean()
    return loss


def isotropy_regularization(embeddings: torch.Tensor) -> torch.Tensor:
    """Penalizes collapse toward a shared centroid by discouraging high cosine similarity to the batch mean."""
    embeddings = F.normalize(embeddings, p=2, dim=1)
    batch_mean = embeddings.mean(dim=0)
    batch_mean = F.normalize(batch_mean, p=2, dim=0)
    cosine_to_mean = (embeddings * batch_mean).sum(dim=1)
    return (cosine_to_mean ** 2).mean()


def kd_gzsl_loss(
    student_embs: torch.Tensor,
    labels: list,
    episode_protos: torch.Tensor,
    proto_labels: list,
    proto_dict: dict,
    temperature: float = 3.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    """
    Meta knowledge distillation loss for GZSL.

    Combines KL divergence against the teacher's soft similarity distribution
    with a hard cross-entropy target.

    Args:
        student_embs: L2-normalized projections from the Student MLP (batch, dim).
        labels: String class labels for each sample in the batch.
        episode_protos: Stacked episode prototypes tensor (num_ep_classes, dim).
        proto_labels: Ordered list of class labels corresponding to episode_protos rows.
        proto_dict: Mapping {label -> numpy prototype array} for teacher logits.
        temperature: Softening temperature T.
        alpha: Weight on distillation loss vs. hard CE.
    """
    device = student_embs.device

    # Teacher soft targets: cosine similarity of the *target* prototype to all episode prototypes
    batch_target_protos = torch.stack([
        torch.tensor(proto_dict[l], dtype=torch.float32) for l in labels
    ]).to(device)
    teacher_logits = torch.matmul(batch_target_protos, episode_protos.T) / temperature
    teacher_probs = F.softmax(teacher_logits, dim=1)

    student_logits = torch.matmul(student_embs, episode_protos.T) / temperature
    student_log_probs = F.log_softmax(student_logits, dim=1)

    distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    hard_targets = torch.tensor([proto_labels.index(l) for l in labels], device=device)
    hard_loss = F.nll_loss(F.log_softmax(student_logits * temperature, dim=1), hard_targets)

    return alpha * distill_loss + (1 - alpha) * hard_loss
