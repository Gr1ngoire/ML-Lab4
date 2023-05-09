import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from PIL import Image

digits = load_digits()

print(digits.DESCR)

print(digits.target[::100])
print(digits.data.shape)
print(digits.target.shape)

# TASK 1
print(digits.images[13])
print(digits.data[13])
figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))
for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)

    
plt.tight_layout()

# TASK 2
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=11, test_size=0.20)

print(X_train.shape)
print(X_test.shape)

# TASK 3
knn = KNeighborsClassifier()
knn.fit(X=X_train, y=y_train)

# TASK 4
predicted = knn.predict(X=X_test)
expected = y_test

# TASK 5
print(predicted[:20])
print(expected[:20])

# TASK 6 
# TASK 6.1
print(f'{knn.score(X_test, y_test):.2%}')

# TASK 6.2
confusion = confusion_matrix(y_true=expected, y_pred=predicted)
print(confusion)

# TASK 7
names = [str(digit) for digit in digits.target_names]
print(classification_report(expected, predicted, target_names=names))

# TASK 8
svc_model = SVC()
svc_model.fit(X=X_train, y=y_train)
svc_predicted = svc_model.predict(X_test)
print(svc_predicted[:20])
print(expected[:20])

naive_bayes = GaussianNB()
naive_bayes.fit(X=X_train, y=y_train)
naive_bayes_predicted = naive_bayes.predict(X_test)
print(naive_bayes_predicted[:20])
print(expected[:20])

# TASK 9
print('Task 9')
k_1_classifier = KNeighborsClassifier(n_neighbors=1)
k_1_classifier.fit(X=X_train, y=y_train)
k_1_predicted = k_1_classifier.predict(X=X_test)
print(k_1_predicted[:20])
print(expected[:20])

k_9_classifier = KNeighborsClassifier(n_neighbors=9)
k_9_classifier.fit(X=X_train, y=y_train)
k_9_predicted = k_9_classifier.predict(X=X_test)
print(k_9_predicted[:20])
print(expected[:20])


# ADDITIONAL

# Loading a 8 x 8 image
img=Image.open('six.png')
data = list(img.getdata())

# Converting pixels values into flat values
pix_val_flat = [int(sum(x) / len(x)) for x in data]

# Convert 0-255 RGB format into 0-16 format
to_black_pattern_format_0_16 = lambda val: math.ceil(abs(val - 255) / 16)

pix_val_flat = [to_black_pattern_format_0_16(x) for x in pix_val_flat]
print(pix_val_flat)

custom_predicted = k_9_classifier.predict([pix_val_flat])
print(f'Predicted: {custom_predicted[0]}')
print(f'Expected 6')