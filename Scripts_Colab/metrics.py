import torch
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from tqdm import tqdm  # Barra de progresso



def calculate_metrics(module, model):
    # Executa o teste para registro das métricas
    steps_outputs_metrics = []
    model.eval()
    
    for images, gt_masks in tqdm(module, desc="Calculando métricas"):
        with torch.no_grad():
            logits = model(images)
        pr_masks = F.softmax(logits, dim=1)
        pr_masks = torch.argmax(pr_masks, dim=1)

        # first compute statistics for true positives, false positives, false negative and
        # true negative "pixels"
        tp, fp, fn, tn = smp.metrics.get_stats(
            gt_masks, pr_masks, mode="multiclass", num_classes=4
        )
        steps_outputs_metrics.append({"tp": tp, "fp": fp, "fn": fn, "tn": tn})


    tp = torch.cat([x["tp"] for x in steps_outputs_metrics])
    fp = torch.cat([x["fp"] for x in steps_outputs_metrics])
    fn = torch.cat([x["fn"] for x in steps_outputs_metrics])
    tn = torch.cat([x["tn"] for x in steps_outputs_metrics])
    
    reductions = ["micro", "macro", "micro-imagewise", "macro-imagewise"]
    
    for reduction in reductions:
        acuracia = smp.metrics.accuracy(tp, fp, fn, tn, reduction=reduction)
        acuracia_balanceada = smp.metrics.balanced_accuracy(
            tp, fp, fn, tn, reduction=reduction
        )
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=reduction)
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction=reduction)
        f2_score = smp.metrics.fbeta_score(
            tp, fp, fn, tn, beta=2, reduction=reduction
        )
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction=reduction)

        print(f"Redução: {reduction}")
        print(f"Acurácia no conjunto de teste: {acuracia:.4f}")
        print(f"Acurácia Balanceada no conjunto de teste: {acuracia_balanceada:.4f}")
        print(f"IoU no conjunto de teste: {iou:.4f}")
        print(f"F1 no conjunto de teste: {f1_score:.4f}")
        print(f"F2 no conjunto de teste: {f2_score:.4f}")
        print(f"Recall no conjunto de teste: {recall:.4f}")
        print("\n")

    return acuracia, acuracia_balanceada, iou, f1_score, f2_score, recall
