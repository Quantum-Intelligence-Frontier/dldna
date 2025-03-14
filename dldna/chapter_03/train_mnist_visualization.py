import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchvision

class ConvNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)    # 28x28 -> 26x26
        self.conv2 = nn.Conv2d(16, 32, 3)   # 13x13 -> 11x11
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(32 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 10)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        self.feat1 = F.relu(self.conv1(x))  # 저장용 중간 특징
        x = F.max_pool2d(self.feat1, 2)
        self.feat2 = F.relu(self.conv2(x))  # 저장용 중간 특징
        x = F.max_pool2d(self.feat2, 2)
        x = x.view(batch_size, -1)
        self.feat3 = F.relu(self.fc1(x))    # 저장용 중간 특징
        x = self.dropout(self.feat3)
        x = self.fc2(x)
        return x

    def get_embeddings(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(batch_size, -1)
        return F.relu(self.fc1(x))

    def get_features(self):
        return self.feat1, self.feat2, self.feat3

class VisualizationUtils:
    @staticmethod
    def make_grid_with_labels(images, labels, num_images=8):
        # 이미지와 레이블을 결합하여 그리드 생성
        fig = plt.figure(figsize=(8, 8))
        for idx in range(min(num_images, len(images))):
            ax = fig.add_subplot(int(np.sqrt(num_images)), int(np.sqrt(num_images)), idx+1)
            ax.imshow(images[idx].squeeze(), cmap='gray')
            ax.set_title(f'Label: {labels[idx]}')
            ax.axis('off')
        return fig

    @staticmethod
    def visualize_feature_maps(feature_maps, num_features=8):
        # 특징 맵 시각화
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = feature_maps.detach().cpu()
        grid = torchvision.utils.make_grid(
            feature_maps[:num_features].unsqueeze(1),
            normalize=True,
            nrow=int(np.sqrt(num_features))
        )
        return grid

    @staticmethod
    def plot_confusion_matrix(cm, class_names):
        # 혼동 행렬 시각화
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        plt.tight_layout()
        return fig
    
def get_embeddings_and_labels(model, loader, device, num_samples=1000):
    embeddings = []
    labels = []
    with torch.no_grad():
        for data, target in loader:
            if len(embeddings) * data.size(0) >= num_samples:
                break
            data = data.to(device)
            embedding = model.get_embeddings(data)
            embeddings.append(embedding.cpu())
            labels.append(target)
    return torch.cat(embeddings), torch.cat(labels)


def train(hparams_dict=None):
    if hparams_dict is None:
        hparams_dict = {
            'batch_size': 32,           
            'learning_rate': 0.0001,    
            'dropout_rate': 0.3,        
            'epochs': 3,
            'optimizer': 'Adam',
            'weight_decay': 1e-5,
            'momentum': 0.9,
            'scheduler_step': 1,
            'scheduler_gamma': 0.1
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_interval = 50  # 더 자주 로깅

    # 데이터 로드
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=hparams_dict['batch_size'], 
                            shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=hparams_dict['batch_size'], 
                           drop_last=True)

    # 모델 설정
    model = ConvNet(dropout_rate=hparams_dict['dropout_rate']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=hparams_dict['learning_rate'],
        weight_decay=hparams_dict['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=hparams_dict['scheduler_step'], 
        gamma=hparams_dict['scheduler_gamma']
    )
    criterion = nn.CrossEntropyLoss()

    # 텐서보드 설정
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', f'mnist_{current_time}')
    writer = SummaryWriter(log_dir)

    # 모델 그래프 기록
    images, _ = next(iter(train_loader))
    writer.add_graph(model, images.to(device))

    # 하이퍼파라미터 기록
    writer.add_hparams(
        hparams_dict,
        {'hparam/accuracy': 0, 'hparam/loss': 0}
    )

    # 학습 루프
    global_step = 0
    best_accuracy = 0
    viz_utils = VisualizationUtils()

    for epoch in range(hparams_dict['epochs']):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                # 1. 기본 메트릭 기록
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                accuracy = correct / len(target)
                
                writer.add_scalar('Metrics/train_loss', loss.item(), global_step)
                writer.add_scalar('Metrics/train_accuracy', accuracy, global_step)
                writer.add_scalar('Learning/learning_rate', 
                                optimizer.param_groups[0]['lr'], global_step)

                # 2. 그래디언트 및 가중치 분포 기록
                for name, param in model.named_parameters():
                    writer.add_histogram(f'Parameters/{name}', param.data, global_step)
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, global_step)
                        writer.add_scalar(f'Gradients/norm_{name}', 
                                        param.grad.norm().item(), global_step)

                # 3. 특징 맵 시각화
                feat1, feat2, feat3 = model.get_features()
                writer.add_image('Features/conv1', 
                               viz_utils.visualize_feature_maps(feat1[0]), global_step)
                writer.add_image('Features/conv2', 
                               viz_utils.visualize_feature_maps(feat2[0]), global_step)

                # 4. 입력 이미지 시각화
                if batch_idx == 0:
                    img_grid = torchvision.utils.make_grid(data[:8])
                    writer.add_image('Images/input', img_grid, global_step)

                    # 컨볼루션 필터 시각화
                    filters = model.conv1.weight.data.cpu()
                    writer.add_image('Filters/conv1', 
                                   torchvision.utils.make_grid(filters), global_step)

                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {loss.item():.6f}\t'
                      f'Accuracy: {accuracy:.4f}')

            global_step += 1

        # 에포크 종료시 검증
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        accuracy = correct / len(test_loader.dataset)

        # 검증 지표 기록
        writer.add_scalar('Metrics/test_loss', test_loss, epoch)
        writer.add_scalar('Metrics/test_accuracy', accuracy, epoch)

        print(f'Epoch {epoch}: Test Loss = {test_loss:.4f}, Accuracy = {accuracy:.4f}')

        # 임베딩 시각화
        embeddings, labels = get_embeddings_and_labels(model, test_loader, device)
        
        # t-SNE로 차원 축소
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings.numpy())
        
        # 임베딩 기록
        label_img = torch.zeros(labels.shape[0], 3, 28, 28)
        for i in range(labels.shape[0]):
            label_img[i] = test_dataset[i][0].repeat(3, 1, 1)
        
        writer.add_embedding(
            embeddings,
            metadata=labels.tolist(),
            label_img=label_img,
            global_step=epoch,
            tag=f'embeddings/epoch_{epoch}'
        )

        scheduler.step()

    # 최종 하이퍼파라미터 메트릭 업데이트
    writer.add_hparams(
        hparams_dict,
        {
            'hparam/accuracy': accuracy,
            'hparam/loss': test_loss
        }
    )

    writer.close()
    return accuracy



if __name__ == '__main__':
    # 단일 실행
    train()

    # 하이퍼파라미터 탐색 실행
    hparams_list = [
        {
            'batch_size': 32, 
            'learning_rate': 0.001, 
            'dropout_rate': 0.3, 
            'epochs': 3,
            'optimizer': 'Adam',
            'weight_decay': 1e-5,
            'momentum': 0.9,
            'scheduler_step': 1,
            'scheduler_gamma': 0.1
        },
        {
            'batch_size': 64, 
            'learning_rate': 0.001, 
            'dropout_rate': 0.5, 
            'epochs': 3,
            'optimizer': 'Adam',
            'weight_decay': 1e-4,
            'momentum': 0.9,
            'scheduler_step': 1,
            'scheduler_gamma': 0.2
        },
        {
            'batch_size': 128, 
            'learning_rate': 0.001, 
            'dropout_rate': 0.7, 
            'epochs': 3,
            'optimizer': 'Adam',
            'weight_decay': 1e-3,
            'momentum': 0.9,
            'scheduler_step': 1,
            'scheduler_gamma': 0.3
        }
    ]
    
    for hparams in hparams_list:
        print(f"\nTraining with hyperparameters: {hparams}")
        train(hparams)
