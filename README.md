# 인공지능 수업 저장소

이 저장소는 대학교 인공지능 수업에서 공부한 것에 대한 내용입니다.

## 인공지능 중간과제
인공지능 수업의 중간 과제로 와인의 퀄리티를 측정하는 인공지능 모델을 만들었습니다.

### 설명
- 언어: Python
- 라이브러리: pandas, scikit-learn (sklearn), Google Colab (주피터 노트북 환경)
- 모델: MLPClassifier (다층 퍼셉트론)
- 평가 및 시각화: confusion_matrix

#### 코드 동작

- sklearn의 라이브러리 import
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
```
- 데이터 로드 후 결측값 처리
```python
train_data = pd.read_csv('C:/Users/ASUS/Documents/winequality_train.csv')
test_data = pd.read_csv('C:/Users/ASUS/Documents/winequality_test.csv')
train_data = train_data.dropna()
test_data = test_data.dropna()
```

- 문자열을 숫자로 변환, 데이터 분
```python
train_data['type'] = train_data['type'].map({'white': 0, 'red': 1})
test_data['type'] = test_data['type'].map({'white': 0, 'red': 1})
X_train = train_data.drop(['quality'], axis=1)
y_train = train_data['quality']
X_test = test_data.drop(['quality'], axis=1)
y_test = test_data['quality']
```
- MLP 모델 생성 및 학습 : 100개의 뉴런을 포함하는 하나의 은닉층과 최대 500번의 반복 훈련이 있는 MLP 모델을 생성
```python
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
mlp.fit(X_train, y_train)
```

- 모델 평가 
```python
print('Accuracy:', mlp.score(X_test[:1000], y_test[:1000]))
accuracy = mlp.score(X_test[:1000], y_test[:1000])
print('Accuracy: {:.2f}%'.format(accuracy * 100))
y_pred = mlp.predict(X_test[:1000])
cm = confusion_matrix(y_test[:1000], y_pred)
print('Confusion Matrix:\n', cm)
```

## 인공지능 기말과제
인공지능 수업의 기말 과제로 꽃 이미지를 분류하는 인공지능 모델을 만들었습니다.

### 기술스택
- TensorFlow Keras
- ResNet50
- Pandas

#### 코드 동작

- 이미지를 로드하고, ResNet50에 입력할 수 있도록 전처리.
```python
x_train, y_train = [], []

for image_path, label in zip(train_images, train_labels):
    img = image.load_img(image_path, target_size=(32, 32))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x_train.append(x)
    y_train.append(label)

x_test, y_test = [], []

for image_path, label in zip(val_images, val_labels):
    img = image.load_img(image_path, target_size=(32, 32))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x_test.append(x)
    y_test.append(label)
```


- ResNet50을 기반으로 하는 모델을 구성하고 Flatten 레이어와 Dense 레이어를 추가하여 완전 연결 신경망을 만듦
- 모델을 컴파일하고 손실 함수, 최적화기, 평가 지표를 설정
```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
cnn = Sequential()
cnn.add(base_model)
cnn.add(Flatten())
cnn.add(Dense(1024, activation='relu'))
cnn.add(Dense(no_class, activation='softmax'))

cnn.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
```


- 모델 학습 및 평가
```python
hits = cnn.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_test, y_test), verbose=1)
res = cnn.evaluate(x_test, y_test, verbose=0)
print("정확도는 ", res[1]*100)
```


-  학습/밸리 데이터로 학습모델을 구축하여 과제 시험 당일 교수님께서 올려주신 테스트 데이터를 가지고 모델을 측정하기 때문에 모델 저장 밎 불러오기 코드가 있습니다.
```python
cnn.save("/content/saved_model.h5")
loaded_model = load_model("/content/saved_model.h5")
```

- 위 모델의 정확률은 86%입니다.
