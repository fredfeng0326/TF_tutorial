from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC(gamma='scale')
clf.fit(X, y)
r = clf.predict([[2., 2.]])
print(r)


# get support vectors
a = clf.support_vectors_


# get indices of support vectors  支持向量在数据集中的索引
b = clf.support_

# get number of support vectors for each class  各类结果中的支持向量的个数
c = clf.n_support_

print(a)
print(b)
print(c)
