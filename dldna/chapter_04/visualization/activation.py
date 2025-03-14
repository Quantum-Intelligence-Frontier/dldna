# import math
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib_inline.backend_inline import set_matplotlib_formats
# set_matplotlib_formats('svg')
# import seaborn as sns

# # 시드값 설정
# np.random.seed(7)
# torch.manual_seed(7)

# def get_gradients(act_func, x):
#     x = x.clone().requires_grad_() # 입력 텐서 객체에 변화를 주지 않기 위해 복사본을 만들고 그래디언트를 활성화 한다.
#     y = act_func(x)
#     y.sum().backward() # backward 연산, 인풋인 x의 그래디언트가 계산된다. 

#     return x.grad 

# def visualize_activations():

#     # 활성 함수를 초기화하고 리스트로 만든다.
#     act_funcs = [act_func() for act_func in act_functions.values()] 

#     ## matplot의 subplot을 생성한다.
#     cols = 3  # 한줄에 3개 그래프를 그린다.
#     rows = math.ceil(len(act_funcs)/cols)
#     fig, axes_list = plt.subplots(rows, cols, figsize=(10, rows*3)) # 서브 플롯을 생성하고 리턴 튜플값을 가져온다 (figure, 좌표값 리스트)
#     fig.subplots_adjust(hspace=0.4) # 그래프 수직으로 간격을 띄운다

#     # 활성함수 딕셔너리 갯수만큼 반복한다.
#     for i, act_func in enumerate(act_funcs):  

#         x = torch.linspace(-4, 4, 200) # 인풋값인 x를 -4~4 범위로 만든다.

#         # 시각화 한다 ===============
#         axes = axes_list[divmod(i,cols)] # 서브 플롯 axes(좌표값)s을 가져온다
        
#         # 개별 서브 플롯별로 설정을 지정한다
#         axes.set_ylim(-1.5, x.max())                # 그래프별 비교가 쉽게 y축 값을 제한한다
#         axes.set(xlabel=None, ylabel=None)          # x, y 라벨(None) 을 제거한다
#         axes.set_title(type(act_func).__name__)     # 활성함수 이름으로 타이틀을 지정한다

#         y = act_func(x) # 활성화 함수 출력값
#         y_grads = get_gradients(act_func, x) # 그래디언트 값
#         x, y, y_grads = x.detach().numpy(), y.detach().numpy(), y_grads.detach().numpy() # detach()로 CPU로 보낸다. (확실하게 하기 위해)

#         sns.lineplot(ax=axes, x=x, y=y, label='Activation') # seaborn lineplot으로 그래프를 그린다. 활성함수
#         sns.lineplot(ax=axes, x=x, y=y_grads, label='Gradient') # 그래디언트 

#     plt.show()

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg')
import seaborn as sns

# Set seed for reproducibility
np.random.seed(7)
torch.manual_seed(7)

def get_gradients(act_func, x):
    """Calculates the gradients of an activation function.

    Args:
        act_func: The activation function (a callable object, e.g., an nn.Module).
        x: The input tensor.

    Returns:
        The gradients of the activation function with respect to x.
    """
    x = x.clone().requires_grad_(True)  # Create a copy and enable gradient tracking.
    y = act_func(x)
    y.sum().backward()  # Perform backward pass; gradients are calculated for x.

    return x.grad

def visualize_activations():
    """Visualizes activation functions and their gradients."""

    # Initialize activation functions and create a list.
    act_funcs = [act_func() for act_func in act_functions.values()]

    # Create matplotlib subplots.
    cols = 3  # 3 graphs per row.
    rows = math.ceil(len(act_funcs) / cols)
    fig, axes_list = plt.subplots(rows, cols, figsize=(10, rows * 3))  # Create subplots and get the return tuple (figure, list of axes)
    fig.subplots_adjust(hspace=0.4)  # Add vertical spacing between graphs

    # Iterate through the activation functions.
    for i, act_func in enumerate(act_funcs):

        x = torch.linspace(-4, 4, 200)  # Create input values x in the range -4 to 4.

        # Visualization ===============================
        axes = axes_list[divmod(i, cols)]  # Get the axes for the subplot

        # Set configurations for each subplot
        axes.set_ylim(-1.5, x.max())  # Limit the y-axis for easier comparison between graphs
        axes.set(xlabel=None, ylabel=None)  # Remove x and y labels (None)
        axes.set_title(type(act_func).__name__)  # Set the title as the name of the activation function

        y = act_func(x)  # Activation function output
        y_grads = get_gradients(act_func, x)  # Gradient values
        x, y, y_grads = x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_grads.detach().cpu().numpy()  # detach() and move to CPU

        sns.lineplot(ax=axes, x=x, y=y, label='Activation')  # Plot the activation function using seaborn's lineplot.
        sns.lineplot(ax=axes, x=x, y=y_grads, label='Gradient')  # Plot the gradient

    plt.show()