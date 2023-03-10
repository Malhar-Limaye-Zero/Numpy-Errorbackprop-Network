import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np
from sklearn.preprocessing import StandardScaler
from network import NN
input_data = pd.read_csv("IRIS_TrainData.csv",header=0)

print("Instances per class: ")
print(input_data["species"].value_counts())
print("\n\n     Bivariate Pairwise relationships between features of train data")
sb.pairplot(input_data, hue="species", height=3, diag_kind="kde")
plt.show()

x_train = []
y_train = []
for i in range(0,len(input_data)):
    x_train.append([input_data.values[i][0],input_data.values[i][1],input_data.values[i][2],input_data.values[i][3]])
    if input_data.values[i][4]=='Iris-setosa':
        y_train.append([0])
    else:
        y_train.append([1])

scale = StandardScaler()
scale.fit(x_train)
xtrain = scale.transform(x_train)
fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(12,5))
ax1.set_title('Before Scaling')
ax2.set_title('After scaling')
print("\n\n\n\n\n\n\n\n   Input values before and after Scaling")
label_names=['sepal_length','sepal_width','petal_length','petal_width']
for i in range(0,4):
    t1=[]
    t2=[]
    for row1 in x_train:
        t1.append(row1[i])
    for row2 in xtrain:
        t2.append(row2[i])
    sb.kdeplot(t1,ax=ax1,label=label_names[i])
    sb.kdeplot(t2,ax=ax2,label=label_names[i])
plt.show()

if __name__ == "__main__":
    nn = NN(x=xtrain,y=np.array(y_train),LR=0.1,epochs=2000)
    nn.train()
    ep=[]
    for i in range (0,len(nn.costlist)):
        ep.append(i)
    plt.plot(ep,nn.costlist,marker = '.')
    plt.xlabel('epoch')
    plt.ylabel('Cost function')
    plt.title("Cost function vs epoch")
    plt.xlim((0,100))
    plt.show()