import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import Adam
import torch

class MyDataset(Dataset):
    def __init__(self, file_path="./ChnSentiCorp_htl_all.csv"):
        super().__init__()
        self.data = pd.read_csv(file_path).dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]

    def __len__(self):
        return len(self.data)

def load_data(file_path="./ChnSentiCorp_htl_all.csv"):
    """加载并返回数据集实例"""
    dataset = MyDataset(file_path)
    print("前5个样本:")
    for i in range(5):
        print(dataset[i])
    return dataset

def split_dataset(dataset, train_ratio=0.9):
    """将数据集划分为训练集和验证集"""
    train_size = int(len(dataset) * train_ratio)
    valid_size = len(dataset) - train_size
    trainset, validset = random_split(dataset, [train_size, valid_size])
    print(f"训练集大小: {len(trainset)}, 验证集大小: {len(validset)}")
    return trainset, validset

def create_dataloaders(trainset, validset, batch_size_train=32, batch_size_valid=64):
    """创建训练集和验证集的数据加载器"""
    tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")

    def collate_func(batch):
        texts, labels = zip(*batch)
        inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs

    trainloader = DataLoader(trainset, batch_size=batch_size_train, shuffle=True, collate_fn=collate_func)
    validloader = DataLoader(validset, batch_size=batch_size_valid, shuffle=False, collate_fn=collate_func)
    return trainloader, validloader, tokenizer  # 新增返回tokenizer
def initialize_model():
    """初始化模型和优化器"""
    model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = Adam(model.parameters(), lr=2e-5)
    return model, optimizer

def evaluate(model, validloader):
    """评估模型性能"""
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    return acc_num / len(validset)

def train(model, optimizer, trainloader, validloader, epochs=3, log_step=100):
    """
    训练模型

    参数:
        model (torch.nn.Module): 模型实例
        optimizer (torch.optim.Optimizer): 优化器实例
        trainloader (DataLoader): 训练集数据加载器
        validloader (DataLoader): 验证集数据加载器
        epochs (int): 训练轮数，默认为3
        log_step (int): 日志打印间隔步数，默认为100
    """

    global_step = 0  # 全局训练步骤计数器

    for epoch in range(epochs):  # 开始训练循环
        print(f"Epoch {epoch + 1}/{epochs}")  # 打印当前轮次信息

        model.train()  # 将模型设置为训练模式
        running_loss = 0.0  # 初始化每轮的累计损失

        for batch in trainloader:  # 迭代训练集中的每个批次
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}  # 如果有GPU可用，将数据移动到GPU

            optimizer.zero_grad()  # 清空之前的梯度

            output = model(**batch)  # 前向传播，计算模型输出
            loss = output.loss  # 获取损失值
            running_loss += loss.item()  # 累计损失

            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数

            if global_step % log_step == 0:  # 按照指定间隔打印日志
                avg_loss = running_loss / (global_step + 1) if global_step > 0 else running_loss
                print(f"Epoch: {epoch}, Global Step: {global_step}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")

            global_step += 1  # 更新全局步骤计数器

        # 每个epoch结束后评估模型性能
        acc = evaluate(model, validloader)  # 调用evaluate函数评估模型在验证集上的准确率
        print(f"Epoch: {epoch}, Accuracy: {acc:.4f}")  # 打印当前轮次的准确率

def predict(model, tokenizer, sentence):
    """对单个句子进行预测"""
    model.eval()
    id2_label = {0: "差评！", 1: "好评！"}
    with torch.inference_mode():
        inputs = tokenizer(sentence, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1)
        print(f"输入：{sentence}\n模型预测结果:{id2_label.get(pred.item())}")

if __name__ == "__main__":
    # 加载数据
    dataset = load_data()
    
    # 划分数据集
    trainset, validset = split_dataset(dataset)
    
    # 创建数据加载器
    trainloader, validloader, tokenizer = create_dataloaders(trainset, validset)
    # 初始化模型和优化器
    model, optimizer = initialize_model()
    
    # 训练模型
    train(model, optimizer, trainloader, validloader)
    
    # 测试预测
    sen = "我觉得这家酒店不错，饭很好吃！"
    predict(model, tokenizer, sen)
