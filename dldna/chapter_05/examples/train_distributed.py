import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_training_dynamics(model, optimizer, train_loader, loss_func, num_epochs, device):
    """
    학습 동안 모델의 동적 특성을 시각화한다.
    각 epoch마다 손실값, 그래디언트 노름, 그리고 초기 파라미터 대비 변화량(L2 norm 합산)을 기록하고,
    이를 3개의 서브플롯(손실, 그래디언트 노름, 파라미터 변화량)으로 시각화한다.
    
    Args:
        model (torch.nn.Module): 학습할 모델.
        optimizer: 모델 파라미터를 업데이트할 최적화 기법.
        train_loader: 학습 데이터 DataLoader.
        loss_func: 손실 함수.
        num_epochs (int): 총 학습 에폭 수.
        device: 계산에 사용할 디바이스 (예: torch.device("cuda") 또는 torch.device("cpu")).
        
    Returns:
        dict: {'loss': [...], 'grad_norm': [...], 'param_change': [...]}
              각 에폭마다 기록된 메트릭 값들.
    """
    model.to(device)
    model.train()

    # 초기 파라미터 복사(파라미터 변화량 측정을 위해)
    initial_params = [param.detach().clone() for param in model.parameters()]

    epoch_losses = []
    epoch_grad_norms = []
    epoch_param_changes = []

    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        running_grad_norm = 0.0
        batch_count = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            optimizer.zero_grad()
            loss.backward()

            # 현재 배치의 모든 파라미터에 대해 그래디언트 노름 합산
            batch_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    batch_grad_norm += param.grad.data.norm(2).item()

            running_grad_norm += batch_grad_norm
            running_loss += loss.item()
            batch_count += 1

            optimizer.step()

        avg_loss = running_loss / batch_count
        avg_grad_norm = running_grad_norm / batch_count

        # 초기 파라미터 대비 변화량(L2 norm 합산)
        total_param_change = 0.0
        for init_param, current_param in zip(initial_params, model.parameters()):
            total_param_change += (current_param.detach() - init_param).norm(2).item()

        epoch_losses.append(avg_loss)
        epoch_grad_norms.append(avg_grad_norm)
        epoch_param_changes.append(total_param_change)

        print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}, Grad Norm: {avg_grad_norm:.4f}, Param Change: {total_param_change:.4f}")

    # 3개의 서브플롯으로 결과 시각화
    epochs = list(range(1, num_epochs + 1))
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].plot(epochs, epoch_losses, marker='o', color='blue')
    axs[0].set_title("Epoch vs Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")

    axs[1].plot(epochs, epoch_grad_norms, marker='o', color='red')
    axs[1].set_title("Epoch vs Gradient Norm")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Gradient Norm")

    axs[2].plot(epochs, epoch_param_changes, marker='o', color='green')
    axs[2].set_title("Epoch vs Parameter Change")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("L2 Norm of Parameter Change")

    plt.tight_layout()
    plt.show()

    metrics = {
        "loss": epoch_losses,
        "grad_norm": epoch_grad_norms,
        "param_change": epoch_param_changes,
    }
    return metrics
