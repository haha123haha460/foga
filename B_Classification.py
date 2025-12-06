import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
from networkIIM import BuildClassfication
from skimage import io
from tqdm import trange
from torch.optim.lr_scheduler import StepLR

class MyDataset(Data.Dataset):
    def __init__(self, df, tokenizer, maxlen, with_labels=True):
        self.tokenizer = tokenizer
        self.with_labels = with_labels
        self.df = df
        self.maxlen = maxlen
        # self.tiffolder=tiffolder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.df.iloc[index]['标签']
        BN = self.df.iloc[index]['对象序号']
        text = ' '.join([str(x) for x in self.df.iloc[index][2:]])
        # print(BN,label,text)

        # imagefile=self.tiffolder+str(BN)+".tif"

        # image = io.imread(imagefile).transpose(2,0,1)

        # image = image.astype(np.float32)

        # image = image*1.0/255.0
        # image = torch.from_numpy(image)



        # print("imagefile",imagefile)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.maxlen,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        token_type_ids = inputs['token_type_ids'].squeeze(0)

        if self.with_labels:
            label = torch.tensor(label, dtype=torch.long)
            return input_ids, attention_mask, token_type_ids,label
        else:
            return input_ids, attention_mask, token_type_ids


if __name__ == '__main__':
	torch.cuda.set_device(0)
	train_df = pd.read_excel('./data/train.xlsx')
	test_df = pd.read_excel('./data/test.xlsx')

	# tiffolder="../data/image/"
	model_name = "bert-base-chinese"
	maxlen = 256
	batch_size = 8
	epoches = 100
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	tokenizer = AutoTokenizer.from_pretrained(model_name)

	train_dataset = MyDataset(train_df, tokenizer, maxlen)
	test_dataset = MyDataset(test_df, tokenizer, maxlen)

	train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

	# fuse_model=WeightedTransformerFusionModel(1000,768,5).cuda()
	fuse_model=BuildClassfication().cuda()


	optimizer = optim.AdamW(fuse_model.parameters(), lr=0.00003, weight_decay=0.0001)
	# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
	
	loss_fn = nn.CrossEntropyLoss()

	# Early stopping parameters
	best_accuracy = 0
	patience = 5
	patience_counter = 0

	train_loss_curve = []
	test_loss_curve = []
	accuracy_curve = []
	confusion_matrices = []

	

	# 训练和测试
	for epoch in trange(epoches):
	    fuse_model.train()
	    sum_loss = 0
	    total_step = len(train_loader)
	    for i, data in enumerate(train_loader):
	        optimizer.zero_grad()
	        data = tuple(p.to(device) for p in data)
	        # image=data[3]


	        pred,_ = fuse_model([data[0], data[1], data[2]])
	        loss = loss_fn(pred, data[3])
	        sum_loss += loss.item()

	        loss.backward()
	        optimizer.step()

	        if (i + 1) % 50 == 0:
	            print(
	                '[Epoch {}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epoches, i + 1, total_step, loss.item()))

	    train_loss_curve.append(sum_loss / total_step)

	    # 在测试集上进行测试
	    fuse_model.eval()
	    test_loss = 0
	    predictions = []
	    true_labels = []
	    with torch.no_grad():
	        for data in test_loader:
	            data = tuple(p.to(device) for p in data)
	            # image=data[3]	       
	            pred,_ = fuse_model([data[0], data[1], data[2]])	            
	            loss = loss_fn(pred, data[3])
	            test_loss += loss.item()
	            pred_labels = pred.argmax(dim=1).cpu().numpy()
	            true_labels.extend(data[3].cpu().numpy())
	            predictions.extend(pred_labels)

	    test_loss_curve.append(test_loss / len(test_loader))
	    accuracy = accuracy_score(true_labels, predictions)
	    accuracy_curve.append(accuracy)
	    confusion_matrices.append(confusion_matrix(true_labels, predictions,labels=[0,1,2,3,4,5,6,7]))
	    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average=None)




	    print('[Epoch {}/{}], Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(epoch + 1, epoches,
	                                                                           test_loss / len(test_loader), accuracy))


	    precision = [round(p, 3) for p in precision]
	    recall = [round(r, 3) for r in recall]
	    f1 = [round(f, 3) for f in f1]
	    print(precision, recall, f1)

	    # 保存整个模型
	    model_save_path = f'./model/model_epoch_{epoch + 1}.pth'
	    torch.save(fuse_model, model_save_path)

	    accuracy_file_path = f'./accuracy/accuracy_epoch.txt'
	    with open(accuracy_file_path, 'a') as f:
	        f.write(f"Accuracy: {accuracy}\n")
	        f.write("Precision per class:\n")
	        f.write(" ".join(map(str, precision)) + "\n")
	        f.write("Recall per class:\n")
	        f.write(" ".join(map(str, recall)) + "\n")
	        f.write("F1-score per class:\n")
	        f.write(" ".join(map(str, f1)) + "\n")
	        f.write("###########################################################################"+"\n")
	    confusion_matrix_file_path = f'./confusion_matrix/confusion_matrix_epoch_{epoch + 1}.txt'
	    os.makedirs(os.path.dirname(confusion_matrix_file_path), exist_ok=True)
	    with open(confusion_matrix_file_path, 'w') as f:
	        f.write("Confusion Matrix:\n")
	        f.write(np.array2string(confusion_matrices[-1], separator=', '))