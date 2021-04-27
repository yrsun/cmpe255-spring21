from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt

def load_data():
    faces = fetch_lfw_people(min_faces_per_person=60)
    return faces

faces = load_data()
X, y = faces.data, faces.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

param_grid = {'C': [1, 5, 10, 50], 'gamma': [0.0001, 0.0005, 0.001, 0.005]}

pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
svc = GridSearchCV(svc, param_grid)
model = make_pipeline(pca, svc)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(metrics.classification_report(y_test, y_pred))

fig = plt.figure()
for i in range(4*6):
    ax = fig.add_subplot(4, 6, i+1, xticks=[], yticks=[])
    ax.imshow(X_test[i].reshape(faces.images[0].shape), cmap=plt.cm.gray)
    color = ('black' if y_pred[i] == y_test[i] else 'red')
    ax.set_title(faces.target[y_pred[i]], fontsize='medium', color=color)
plt.show()

plt.figure()
sn.heatmap(metrics.confusion_matrix(y_pred, y_test))
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()