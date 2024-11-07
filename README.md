import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 학습 데이터 정의
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

# 학습률 0.01
print("[1] 학습률이 0.01인 훈련")
model_01 = nn.Linear(1, 1)
optimizer_01 = optim.SGD(model_01.parameters(), lr=0.01)

for epoch in range(100):
    pred = model_01(x)
    loss = F.mse_loss(pred, y)
    optimizer_01.zero_grad()
    loss.backward()
    optimizer_01.step()

    if (epoch+1) % 20 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

print("\n")  # 결과를 구분하기 위해 빈 줄 출력
