"""
1、先错位训练
2、答问SFT
"""
from torch.utils.data import DataLoader
from divgpt import DivGPT
from torch.optim import Adam

from mydataset import *

def main():
    #错位训练集
    offset_dataset = OffsetDataSet(r'data/offset_context.txt', r'data/chars.txt')
    data_loader = DataLoader(offset_dataset, batch_size=4, shuffle=True, num_workers=1)

    epochs = 10
    lr = 0.01
    model = DivGPT(len(offset_dataset.vocab), 1024, 12, 768, 1, -100)
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        loss_list = []
        for batch_idx, data in enumerate(data_loader):
            x, y = data
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print(f'epoch{epoch + 1}, loss:', sum(loss_list) / len(loss_list))

    torch.save(model, r'model/full_model.pth')
    print("全量模型已保存至 model/full_model.pth")

if __name__ == '__main__':
    main()